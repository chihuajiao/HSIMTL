import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3_gn_relu(in_channel, out_channel, num_group):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 3, 1, 1),
        nn.GroupNorm(num_group, out_channel),
        nn.ReLU(inplace=True),
    )

def downsample2x(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 3, 2, 1),
        nn.ReLU(inplace=True)
    )

class SimpleSEBlock(nn.Module):
    def __init__(self, channel, reduction):
        super(SimpleSEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

def repeat_block(block_channel, r, n):
    layers = [
        nn.Sequential(
            SimpleSEBlock(block_channel, r),
            conv3x3_gn_relu(block_channel, block_channel, r)
        )
        for _ in range(n)]
    return nn.Sequential(*layers)

class FreeNet(nn.Module):
    def __init__(self, config):
        super(FreeNet, self).__init__()
        r = int(16 * config['reduction_ratio'])
        block_channels = [int(bc * config['reduction_ratio'] / r) * r for bc in config['block_channels']]
        
        self.feature_ops = nn.ModuleList([
            conv3x3_gn_relu(config['in_channels'], block_channels[0], r)
        ])
        
        for i in range(4):
            self.feature_ops.append(repeat_block(block_channels[i], r, config['num_blocks'][i]))
            self.feature_ops.append(nn.Identity())
            if i < 3:
                self.feature_ops.append(downsample2x(block_channels[i], block_channels[i + 1]))
        
        inner_dim = int(config['inner_dim'] * config['reduction_ratio'])
        self.reduce_1x1convs = nn.ModuleList([nn.Conv2d(bc, inner_dim, 1) for bc in block_channels])
        self.fuse_3x3convs = nn.ModuleList([nn.Conv2d(inner_dim, inner_dim, 3, 1, 1) for _ in range(4)])
        self.cls_pred_conv = nn.Conv2d(inner_dim, config['num_classes'], 1)
        
    def forward(self, x):
        feat_list = []
        for op in self.feature_ops:
            x = op(x)
            if isinstance(op, nn.Identity):
                feat_list.append(x)
        
        inner_feat_list = [conv(feat) for conv, feat in zip(self.reduce_1x1convs, feat_list)]
        inner_feat_list.reverse()
        
        out_feat_list = [self.fuse_3x3convs[0](inner_feat_list[0])]
        for i in range(1, len(inner_feat_list)):
            top2x = F.interpolate(out_feat_list[-1], size=inner_feat_list[i].size()[2:], mode='nearest')#F.interpolate(out_feat_list[-1], scale_factor=2.0, mode='nearest')
            inner = top2x + inner_feat_list[i]
            out = self.fuse_3x3convs[i](inner)
            out_feat_list.append(out)
        
        final_feat = out_feat_list[-1]
        
        logit = self.cls_pred_conv(final_feat)
        return logit
        # return torch.softmax(logit, dim=1) 

if __name__ == "__main__":
    config = {
        'in_channels': 200,
        'num_classes': 16,
        'block_channels': [64, 128, 256, 512],
        'num_blocks': [2, 2, 2, 2],
        'inner_dim': 128,
        'reduction_ratio': 0.5
    }
    
    
    model = FreeNet(config)
    input_tensor = torch.randn(1, 200, 145, 145)
    from thop import profile
    #145, 145,  200 IP
    #349, 1905, 144 HU
    #1280, 307, 191 DC
    #1723, 476, 244 BE
    output = model(input_tensor)
    flops, params = profile(model, inputs=(input_tensor, ))
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')


    print(output.shape)

    import time

    # output,abu,hsi_re = model(inputs)

    # print(output.shape)  
    # print(abu.shape) 
    # print(hsi_re.shape) 
    def measure_latency(model, inputs, device="cuda:0", runs=50, warmup=10):
        model.eval()
        inputs = inputs.to(device)
        model.to(device)

        # 预热
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(inputs)

        torch.cuda.synchronize() if "cuda" in device else None
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(runs):
                _ = model(inputs)
        torch.cuda.synchronize() if "cuda" in device else None
        end = time.perf_counter()

        avg_latency = (end - start) / runs
        print(f"Latency on {device}: {avg_latency*1000:.2f} ms")
        return avg_latency

    def measure_memory(model, inputs, device="cuda:0"):
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            _ = model(inputs.to(device))
        peak = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"Peak Memory on {device}: {peak:.2f} MB")
        return peak
    
    def measure_throughput(model, inputs, device="cuda:0", runs=50):
        model.eval()
        inputs = inputs.to(device)
        model.to(device)

        torch.cuda.synchronize() if "cuda" in device else None
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(runs):
                _ = model(inputs)
        torch.cuda.synchronize() if "cuda" in device else None
        end = time.perf_counter()

        total_time = end - start
        throughput = (inputs.size(0) * runs) / total_time
        print(f"Throughput on {device}: {throughput:.2f} samples/sec")
        return throughput

    from thop import profile
    #145, 145,  200 IP
    #349, 1905, 144 HU
    #1280, 307, 191 DC
    #1723, 476, 244 BE
    flops, params = profile(model, inputs=(input_tensor, ))
    params = params/1000**2
    flops = flops/1000**3
    print("%.2fM" % params)
    print("%.2fG" % flops)

    # 测试延迟
    measure_latency(model, input_tensor, device="cpu")
    measure_latency(model, input_tensor, device="cuda:0")
    # 测试显存峰值
    measure_memory(model, input_tensor, device="cuda:0")
    # 测试吞吐量
    measure_throughput(model, input_tensor, device="cuda:0")