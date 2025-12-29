# code from SSFCN
# https://github.com/YonghaoXu/SSFCN/blob/master/SSFCN.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SSFCN(nn.Module):
    def __init__(self, num_bands=103, num_classes=9):
        super(SSFCN, self).__init__()
        
        # Spectral分支
        self.spectral_conv1 = nn.Conv2d(num_bands, 64, 1)
        self.spectral_conv2 = nn.Conv2d(64, 64, 1)
        self.spectral_conv3 = nn.Conv2d(64, 64, 1)
        
        # Spatial分支
        self.spatial_dr_conv = nn.Conv2d(num_bands, 64, 1)
        self.spatial_conv1 = nn.Conv2d(64, 64, 3, padding=2, dilation=2)
        self.spatial_conv2 = nn.Conv2d(64, 64, 3, padding=2, dilation=2)
        
        # 自适应池化保持尺寸
        self.avg_pool = nn.AdaptiveAvgPool2d((None, None))  # 保持原始尺寸
        
        # 加权参数
        self.w_spatial = nn.Parameter(torch.Tensor([1.5]))
        self.w_spectral = nn.Parameter(torch.Tensor([1.5]))
        
        # 最终卷积层
        self.final_conv = nn.Conv2d(64, num_classes, 1)
        
                    
    def forward(self, x):

        # Spectral分支
        s1 = F.relu(self.spectral_conv1(x))
        s2 = F.relu(self.spectral_conv2(s1))
        s3 = F.relu(self.spectral_conv3(s2))
        spectral_out = s1 + s2 + s3
        
        # Spatial分支
        d1 = F.relu(self.spatial_dr_conv(x))
        
        # 第一层
        s_conv1 = F.relu(self.spatial_conv1(d1))
        s_pool1 = self.avg_pool(s_conv1)  # 保持尺寸
        
        # 第二层
        s_conv2 = F.relu(self.spatial_conv2(s_pool1))
        s_pool2 = self.avg_pool(s_conv2)  # 保持尺寸
        
        spatial_out = d1 + s_pool1 + s_pool2
        
        # 特征融合
        fused = self.w_spatial * spatial_out + self.w_spectral * spectral_out
        
        # 最终输出
        out = self.final_conv(fused)
        return out  # 转换回[N, H, W, C]

# 验证网络结构
if __name__ == "__main__":
    # 输入尺寸 [1, 610, 340, 103]
    input= torch.randn(1, 244, 1723, 476)
    
    model = SSFCN(num_bands=244, num_classes=8)
    # output = model(input_tensor)
    # print("输入尺寸:", input_tensor.shape)
    # print("输出尺寸:", output.shape)  # 应该得到 [1, 610, 340, 9]
    from thop import profile
    #145, 145,  200 IP
    #349, 1905, 144 HU
    #1280, 307, 191 DC
    #1723, 476, 244 BE
    flops, params = profile(model, inputs=(input, ))
    params = params/1000**2
    flops = flops/1000**3
    print("%.2fM" % params)
    print("%.2fG" % flops)

    # # 验证残差连接尺寸
    # x = torch.randn(1, 103, 610, 340)
    
    # # Spectral分支验证
    # s1 = model.spectral_conv1(x)
    # assert s1.shape == (1, 64, 610, 340)
    
    # # Spatial分支验证
    # d1 = model.spatial_dr_conv(x)
    # s_conv1 = model.spatial_conv1(d1)
    # assert s_conv1.shape == (1, 64, 610, 340)
    
    # s_pool1 = model.avg_pool(s_conv1)
    # assert s_pool1.shape == (1, 64, 610, 340)