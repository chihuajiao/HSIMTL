import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from models.backbone.swimtransformer import SwinTransformer
from models.PPM import PPMHEAD

from torch.utils.tensorboard import SummaryWriter
from models.PPM import PPMHEAD


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)
     
def seg_padding(im, patch_size, fill_value=0):
    # make the image sizes divisible by patch_size
    H, W = im.size(2), im.size(3)
    pad_h, pad_w = 0, 0
    if H % patch_size > 0:
        pad_h = patch_size - (H % patch_size)
    if W % patch_size > 0:
        pad_w = patch_size - (W % patch_size)
    im_padded = im
    if pad_h > 0 or pad_w > 0:
        im_padded = F.pad(im, (0, pad_w, 0, pad_h), value=fill_value)
    return im_padded

def seg_unpadding(y, target_size):
    H, W = target_size
    H_pad, W_pad = y.size(2), y.size(3)
    extra_h = H_pad - H
    extra_w = W_pad - W
    if extra_h > 0:
        y = y[:, :, :-extra_h]
    if extra_w > 0:
        y = y[:, :, :, :-extra_w]
    return y

def get_3d_sp_pos_encoding(B, H, W, device='cuda'):
        x, y = torch.meshgrid(
            torch.linspace(0, 1, H, device=device),
            torch.linspace(0, 1, W, device=device),
            indexing='ij'
        )
        z = (x + y) / 2

        sin_x = torch.sin(torch.pi * 2* x)      # [H, W]
        cos_y = torch.cos(torch.pi *2* y)      # [H, W]
        lin_z = z                             # [H, W]

        pos = torch.stack([sin_x, cos_y, lin_z], dim=0)  # [3, H, W]
        pos = pos.unsqueeze(0).expand(B, -1, -1, -1)     # [B, 3, H, W]
        return pos

def get_3d_ca_pos_encoding(B, H, W, device='cuda'):
        x, y = torch.meshgrid(
            torch.linspace(0, 1, H, device=device),
            torch.linspace(0, 1, W, device=device),
            indexing='ij'
        )

        lin_z = torch.sin(torch.pi * 2*x)/2 +  torch.cos(torch.pi *2* y)/2                            # [H, W]

        pos = torch.stack([x, y, lin_z], dim=0)  # [3, H, W]
        pos = pos.unsqueeze(0).expand(B, -1, -1, -1)     # [B, 3, H, W]
        return pos

class SpatialLinearAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q_proj = nn.Conv2d(channels+3, channels, 1)
        self.k_proj = nn.Conv2d(channels, channels, 1)
        self.v_proj = nn.Conv2d(channels, channels, 1)
        
    def forward(self, q, kv):

        B, C, H, W = q.shape
        coords = get_3d_sp_pos_encoding(B, H, W, device=q.device)
        q_in = torch.cat([q, coords], dim=1)
        
        Q = self.q_proj(q_in)  # [B,C,H,W]
        K = self.k_proj(kv).mean(-1, keepdim=True)  # [B,C,H,1]
        V = self.v_proj(kv).mean(-2, keepdim=True)  # [B,C,H,1]

        attn_scores = torch.tanh((Q * K))  # [B,1,H,W]
        out = attn_scores * V  # [B,C,H,W]

        return q + out

    

class ChannelLinearAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q_proj = nn.Conv2d(channels+3, channels, 1)
        self.k_proj = nn.Conv2d(channels, channels, 1)
        self.v_proj = nn.Conv2d(channels, channels, 1)

    def forward(self, q, kv):
        B, C, H, W = q.shape
        coords = get_3d_ca_pos_encoding(B, H, W, device=q.device)
        q_in = torch.cat([q, coords], dim=1)
        Q = self.q_proj(q_in)  # [B,C,H,W]
        K = self.k_proj(kv).mean(1, keepdim=True)  # [B,1,H,W]
        V = self.v_proj(kv).mean(1, keepdim=True)  # [B,1,H,W]
        
        attn_scores = torch.tanh((Q * K))  # [B,C,1,1]
        out = attn_scores * V  # [B,C,H,W]

        return q + out

class ALAKT(nn.Module):
    def __init__(self, channels: int, num_features: int, reduction: int = 16):
        super().__init__()
        self.num_features = num_features
        
        self.all_score = nn.ModuleList([
        nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(channels, channels // 4, 1),
        nn.LayerNorm([channels // 4, 1, 1]),
        nn.GELU(),
        nn.Conv2d(channels // 4, channels // 16, 1),
        nn.LayerNorm([channels // 16, 1, 1]),
        nn.GELU(),
        nn.Conv2d(channels // 16, 1, 1),
        ) for _ in range(num_features)
        ])

        self.register_buffer('shapley_weight', torch.zeros(num_features))


    def forward(self, features):
        assert len(features) == self.num_features
        B = features[0].size(0)
        scores = []
        for i in range(self.num_features):
            score = self.all_score[i](features[i])  
            scores.append(score.view(B, 1))

        all_scores = torch.cat(scores, dim=1)
        weights = F.softmax(all_scores, dim=1) 
        self.shapley_weight.copy_(weights.mean(0).detach())
        fused = sum(weights[:, i].view(B, 1, 1, 1) * features[i]
                    for i in range(self.num_features))
        return fused, weights


class FPNHEAD_two(nn.Module):
    def __init__(self, channels=768, out_channels=256):
        super(FPNHEAD_two, self).__init__()

        self.PPMHead = PPMHEAD(in_channels=channels, out_channels=out_channels)
        self.Conv_fuse1 = nn.Sequential(
            nn.Conv2d(channels//2, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.Conv_fuse1_ = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.Conv_fuse2 = nn.Sequential(
            nn.Conv2d(channels//4, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )    
        self.Conv_fuse2_ = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        self.Conv_fuse3 = nn.Sequential(
            nn.Conv2d(channels//8, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ) 
        self.Conv_fuse3_ = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
      
        self.conv_x1 = nn.Conv2d(out_channels, out_channels, 1)

        self.Conv_fuse11_ = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
  
        self.Conv_fuse22_ = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.Conv_fuse33_ = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.fuse_all1 = nn.Sequential(
            nn.Conv2d(out_channels*1, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
        self.fuse_all2 = nn.Sequential(
            nn.Conv2d(out_channels*1, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.conv_x11 = nn.Conv2d(out_channels, out_channels, 1)

        self.um_fuse0 = SpatialLinearAttention(out_channels)
        self.um_fuse1 = SpatialLinearAttention(out_channels)
        self.um_fuse2 = SpatialLinearAttention(out_channels)
        self.um_fuse3 = SpatialLinearAttention(out_channels)

        self.cls_fuse0 = ChannelLinearAttention(out_channels)
        self.cls_fuse1 = ChannelLinearAttention(out_channels)
        self.cls_fuse2 = ChannelLinearAttention(out_channels)
        self.cls_fuse3 = ChannelLinearAttention(out_channels)

        self.ALAKT_cls = ALAKT(channels=out_channels, num_features=8)
        self.ALAKT_um = ALAKT(channels=out_channels, num_features=8)

    def forward(self, input_fpn):

        x1 = self.PPMHead(input_fpn[-1]) 

        x1_cls =  self.cls_fuse0(x1,x1)  
        x1_um =  self.um_fuse0(x1,x1)   
        
        x = F.interpolate(x1_cls, scale_factor=2, mode='bilinear', align_corners=False)
        xx = F.interpolate(x1_um, scale_factor=2, mode='bilinear', align_corners=False)
        

        q = self.Conv_fuse1(input_fpn[-2]) 
        kv = self.conv_x1(x) 
        kv1 = self.conv_x11(xx)
        x =  self.cls_fuse1(q,q)  +kv
        xx =  self.um_fuse1(q,q)  +kv1  
    
        x2 = self.Conv_fuse1_(x)
        x = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
        xx2 = self.Conv_fuse11_(xx)
        xx = F.interpolate(xx2, scale_factor=2, mode='bilinear', align_corners=False)

        q = self.Conv_fuse2(input_fpn[-3]) 
        kv = self.conv_x1(x) 
        kv1 = self.conv_x11(xx)
        x =  self.cls_fuse2(q,q) +kv
        xx =  self.um_fuse2(q,q) +kv1
         
        x3 = self.Conv_fuse2_(x)
        x = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        xx3 = self.Conv_fuse22_(xx) 
        xx = F.interpolate(xx3, scale_factor=2, mode='bilinear', align_corners=False)

        q = self.Conv_fuse3(input_fpn[-4]) 
        kv = self.conv_x1(x) 
        kv1 = self.conv_x11(xx)
        x =  self.cls_fuse3(q,q) +kv
        xx =  self.um_fuse3(q,q) +kv1

        x4 = self.Conv_fuse3_(x)
        xx4 = self.Conv_fuse33_(xx)
        
        x1_cls = F.interpolate(x1_cls, scale_factor=8, mode='bilinear', align_corners=False)
        x2 = F.interpolate(x2, scale_factor=4, mode='bilinear', align_corners=False)
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)

        x1_um = F.interpolate(x1_um, scale_factor=8, mode='bilinear', align_corners=False)
        xx2 = F.interpolate(xx2, scale_factor=4, mode='bilinear', align_corners=False)
        xx3 = F.interpolate(xx3, scale_factor=2, mode='bilinear', align_corners=False)

        x_cls, shapley_weights_cls = self.ALAKT_cls([x1_cls, x2, x3, x4,x1_um,xx2, xx3, xx4])
        x_um, shapley_weights_um = self.ALAKT_um([x1_cls, x2, x3, x4,x1_um,xx2, xx3, xx4])

        x_cls = self.fuse_all1(x_cls)
        x_um = self.fuse_all2(x_um)

        return x_cls,x_um

class MSUANet(nn.Module):
    def __init__(self, num_classes,in_channel):
        super(MSUANet, self).__init__()
        P = num_classes
        L = in_channel
        self.patch_size = 32
        self.num_classes = num_classes
        self.in_channels = 96 * 8
        self.channels = 256

        self.backbone = SwinTransformer(
            in_chans=L,
            patch_size=4,
            window_size=7,
            embed_dim=96,
            depths=(2, 2, 6, 2),
            num_heads=(3, 6, 12, 24),
            num_classes=num_classes
        )

        self.decoder_two = FPNHEAD_two(channels=96 * 8, out_channels=256)


        self.cls_seg = nn.Sequential(
            nn.Conv2d(self.channels, self.num_classes, kernel_size=3, padding=1),
        )
        
        self.conv_2d_HU = nn.Sequential(    
            nn.Conv2d(256, 128, kernel_size=(1,1), stride=1, padding=(0,0)),
            nn.BatchNorm2d(128,momentum=0.9),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, P, kernel_size=(1,1), stride=1, padding=(0,0)),
            nn.BatchNorm2d(P,momentum=0.9),
            # nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.decoder1 = nn.Sequential(
            nn.Conv2d(P, L, kernel_size=1, stride=1, bias=False),
            nn.ReLU(),
        )
        
    def forward(self, x):
        H_ori, W_ori = x.size(2), x.size(3)

        x = seg_padding(x, self.patch_size)
        x = self.backbone(x)  
        x_cls,x_um= self.decoder_two(x)
        x_cls = F.interpolate(x_cls, scale_factor=4, mode='bilinear', align_corners=False)
        x_um = F.interpolate(x_um, scale_factor=4, mode='bilinear', align_corners=False)
        x_cls1 = seg_unpadding(x_cls, (H_ori, W_ori))
        x_um = seg_unpadding(x_um, (H_ori, W_ori))

        abu = self.conv_2d_HU(x_um)
        hsi_re = self.decoder1(abu)
        x_cls = self.cls_seg(x_cls1)
 
        return x_cls, abu, hsi_re
        
    
if __name__ == '__main__':

    
    inputs = torch.rand(1,200, 145,145).to('cuda:0') 
    model = MSUANet(num_classes=16,in_channel=200).to('cuda:0') 

    output,abu,hsi_re = model(inputs)
    print(output.shape)  
    print(abu.shape) 
    print(hsi_re.shape) 
    