"""将DeepPhys PyTorch模型转换为ONNX格式 (支持NHWC输入)"""

import torch
import torch.nn as nn
import os
from models.deepphys_model import DeepPhys


class DeepPhysNHWCWrapper(nn.Module):
    """
    包装DeepPhys模型，使其接受NHWC输入并转换为NCHW供内部使用
    输入: (Batch, Height, Width, Channels) -> 6 channels
    输出: (Batch, 1)
    """
    def __init__(self, original_model):
        super(DeepPhysNHWCWrapper, self).__init__()
        self.model = original_model

    def forward(self, x):
        # x shape: [N, H, W, C] (NHWC)
        # 转换为 [N, C, H, W] (NCHW)
        x_nchw = x.permute(0, 3, 1, 2)
        
        # 原始模型前向传播
        out = self.model(x_nchw)
        return out


def convert_to_onnx(model_path, output_path=None, device='cpu', img_size=72):
    """
    将DeepPhys模型从PyTorch格式转换为ONNX格式 (NHWC输入)
    
    Args:
        model_path: PyTorch模型文件路径 (.pth)
        output_path: ONNX输出文件路径 (默认与模型同目录)
        device: 运行设备 (cpu/cuda)
        img_size: 图像尺寸 (36/72/96)
    
    Returns:
        output_path: 生成的ONNX文件路径
    """
    # 设置设备
    device = torch.device(device)
    print(f"使用设备: {device}")
    
    # 1. 创建原始DeepPhys模型 (NCHW内部逻辑)
    print("正在创建DeepPhys模型...")
    original_model = DeepPhys(
        in_channels=3,
        nb_filters1=32,
        nb_filters2=64,
        kernel_size=3,
        dropout_rate1=0.25,
        dropout_rate2=0.5,
        pool_size=(2, 2),
        nb_dense=128,
        img_size=img_size
    )
    
    # 加载模型权重
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    print(f"正在加载模型权重: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # 处理不同的checkpoint格式
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # 加载状态字典
    original_model.load_state_dict(state_dict, strict=False)
    original_model.to(device)
    original_model.eval()  # 设置为评估模式
    
    # 2. 包装模型以支持NHWC输入
    wrapped_model = DeepPhysNHWCWrapper(original_model)
    wrapped_model.to(device)
    wrapped_model.eval()
    
    print("模型加载并包装成功! (输入格式: NHWC)")
    
    # 3. 创建示例输入 (NHWC格式: batch_size, height, width, channels)
    # 注意：DeepPhys需要6个通道 (3 diff + 3 raw)
    dummy_input_nhwc = torch.randn(1, img_size, img_size, 6).to(device)
    print(f"示例输入形状 (NHWC): {dummy_input_nhwc.shape}")
    
    # 确定输出路径
    if output_path is None:
        base_name = os.path.splitext(model_path)[0]
        output_path = f"{base_name}_nhwc.onnx"
    
    # 执行转换
    print(f"正在转换为ONNX格式 (NHWC输入)...")
    print(f"输出路径: {output_path}")
    
    # 禁用动态轴，固定batch_size=1（项目仅使用单帧推理）
    # 注意：不使用dynamic_axes参数，所有维度都将固定
    
    try:
        torch.onnx.export(
            wrapped_model,                # 包装后的模型
            dummy_input_nhwc,             # NHWC示例输入
            output_path,                  # 输出文件路径
            export_params=True,           # 导出训练好的参数
            opset_version=11,             # ONNX算子集版本
            do_constant_folding=True,     # 是否执行常量折叠优化
            input_names=['input'],        # 输入层名称
            output_names=['output'],      # 输出层名称
            # dynamic_axes已移除，所有维度固定为静态
            verbose=False                 # 是否打印详细信息
        )
        
        print(f"✓ 转换成功! ONNX模型已保存至: {output_path}")
        print(f"  - 输入形状: [1, {img_size}, {img_size}, 6] (NHWC, 固定batch_size)")
        print(f"  - 输出形状: [1, 1]")
        
        # 验证ONNX模型
        verify_onnx_model(output_path, dummy_input_nhwc, wrapped_model, device, is_nhwc=True)
        
        return output_path
        
    except Exception as e:
        print(f"✗ 转换失败: {str(e)}")
        raise


def verify_onnx_model(onnx_path, dummy_input, original_model, device, is_nhwc=False):
    """
    验证ONNX模型的正确性
    
    Args:
        onnx_path: ONNX模型路径
        dummy_input: 测试输入
        original_model: 原始PyTorch模型 (如果是NHWC wrapper，这里传wrapper)
        device: 运行设备
        is_nhwc: 输入是否为NHWC格式
    """
    try:
        import onnx
        import onnxruntime as ort
        
        print("\n正在验证ONNX模型...")
        
        # 加载并检查ONNX模型
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX模型结构检查通过")
        
        # 获取PyTorch模型的输出
        with torch.no_grad():
            pytorch_output = original_model(dummy_input).cpu().numpy()
        
        # 获取ONNX模型的输出
        # 注意：BPU通常使用CPU EP进行验证即可，或者指定特定的EP
        ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        input_name = ort_session.get_inputs()[0].name
        ort_inputs = {input_name: dummy_input.cpu().numpy()}
        onnx_output = ort_session.run(None, ort_inputs)[0]
        
        # 比较输出差异
        diff = abs(pytorch_output - onnx_output).max()
        print(f"✓ PyTorch与ONNX输出最大差异: {diff:.2e}")
        
        if diff < 1e-4:
            print("验证通过: ONNX模型与PyTorch模型输出一致")
        else:
            print("警告: 输出存在较大差异，请检查模型转换")
            
    except ImportError:
        print("跳过验证: 未安装onnx或onnxruntime库")
        print("提示: 安装命令 pip install onnx onnxruntime")
    except Exception as e:
        print(f"验证过程出错: {str(e)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='DeepPhys模型转换为ONNX格式 (NHWC输入)')
    parser.add_argument('--model_path', type=str, 
                       default='models/SCAMPS_DeepPhys.pth',
                       help='PyTorch模型文件路径')
    parser.add_argument('--output_path', type=str, default=None,
                       help='ONNX输出文件路径 (可选)')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='运行设备')
    parser.add_argument('--img_size', type=int, default=72,
                       choices=[36, 72, 96],
                       help='图像尺寸')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DeepPhys PyTorch → ONNX 模型转换器 (NHWC输入版)")
    print("=" * 60)
    
    try:
        convert_to_onnx(
            model_path=args.model_path,
            output_path=args.output_path,
            device=args.device,
            img_size=args.img_size
        )
        print("\n" + "=" * 60)
        print("转换完成!")
        print("=" * 60)
    except Exception as e:
        print(f"\n转换失败: {str(e)}")
        exit(1)