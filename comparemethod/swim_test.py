import torch
from SwimUnet import SwinUnet  # 替换为你的模型文件路径
from SwimUnet_config import get_config
import argparse

def test_model():
    # 1. 创建模拟配置参数
    args = argparse.Namespace()
    args.batch_size = 1
    args.zip = None
    args.cache_mode = True
    args.resume = None
    args.cfg = 'swin_tiny_patch4_window7_224_lite.yaml'  # 你的配置文件路径
    args.opts = [
        'MODEL.SWIN.EMBED_DIM', '96',
        'MODEL.SWIN.DEPTHS', '[2,2,2,2]',
        'MODEL.SWIN.NUM_HEADS', '[3,6,12,24]',
        'DATA.IMG_SIZE', '224',
        'MODEL.NUM_CLASSES', '9'
    ]
    
    # 2. 初始化模型
    config = get_config(args)
    model = SwinUnet(config, img_size=224, num_classes=9)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # 3. 测试标准输入
    with torch.no_grad():
        # 测试三通道输入
        x = torch.randn(2, 3, 224, 224).to(device)  # (batch, channel, H, W)
        output = model(x)
        print("输入形状:", x.shape)
        print("输出形状:", output.shape)
        assert output.shape == (2, 9, 224, 224), f"形状错误，预期 (2,9,224,224)，实际 {output.shape}"

        # 测试单通道自动扩展
        x = torch.randn(2, 1, 224, 224).to(device)
        output = model(x)
        print("\n单通道输入自动扩展后的输出形状:", output.shape)
        assert output.shape == (2, 9, 224, 224), f"形状错误，预期 (2,9,224,224)，实际 {output.shape}"

    print("\n基本测试通过！")

if __name__ == "__main__":
    test_model()