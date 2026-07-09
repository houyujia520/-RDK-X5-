"""DeepPhys模型加载和推理模块"""

import torch
import torch.nn as nn
import os


class AttentionMask(nn.Module):
    """注意力掩码层"""
    
    def __init__(self):
        super(AttentionMask, self).__init__()
    
    def forward(self, x):
        xsum = torch.sum(x, dim=2, keepdim=True)
        xsum = torch.sum(xsum, dim=3, keepdim=True)
        xshape = tuple(x.size())
        return x / xsum * xshape[2] * xshape[3] * 0.5


class DeepPhys(nn.Module):
    """DeepPhys模型架构"""
    
    def __init__(self, in_channels=3, nb_filters1=32, nb_filters2=64, 
                 kernel_size=3, dropout_rate1=0.25, dropout_rate2=0.5, 
                 pool_size=(2, 2), nb_dense=128, img_size=72):
        """
        DeepPhys模型初始化
        
        Args:
            in_channels: 输入通道数
            nb_filters1: 第一层卷积核数量
            nb_filters2: 第二层卷积核数量
            kernel_size: 卷积核大小
            dropout_rate1: 第一个dropout率
            dropout_rate2: 第二个dropout率
            pool_size: 池化核大小
            nb_dense: 全连接层神经元数
            img_size: 图像尺寸
        """
        super(DeepPhys, self).__init__()
        
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.pool_size = pool_size
        self.nb_filters1 = nb_filters1
        self.nb_filters2 = nb_filters2
        self.nb_dense = nb_dense
        
        # Motion branch convolutions
        self.motion_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, 
                                     kernel_size=self.kernel_size, 
                                     padding=(1, 1), bias=True)
        self.motion_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, 
                                     kernel_size=self.kernel_size, bias=True)
        self.motion_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters2, 
                                     kernel_size=self.kernel_size, 
                                     padding=(1, 1), bias=True)
        self.motion_conv4 = nn.Conv2d(self.nb_filters2, self.nb_filters2, 
                                     kernel_size=self.kernel_size, bias=True)
        
        # Appearance branch convolutions
        self.appearance_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, 
                                         kernel_size=self.kernel_size,
                                         padding=(1, 1), bias=True)
        self.appearance_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, 
                                         kernel_size=self.kernel_size, bias=True)
        self.appearance_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters2, 
                                         kernel_size=self.kernel_size,
                                         padding=(1, 1), bias=True)
        self.appearance_conv4 = nn.Conv2d(self.nb_filters2, self.nb_filters2, 
                                         kernel_size=self.kernel_size, bias=True)
        
        # Attention layers
        self.appearance_att_conv1 = nn.Conv2d(self.nb_filters1, 1, 
                                             kernel_size=1, 
                                             padding=(0, 0), bias=True)
        self.attn_mask_1 = AttentionMask()
        self.appearance_att_conv2 = nn.Conv2d(self.nb_filters2, 1, 
                                             kernel_size=1, 
                                             padding=(0, 0), bias=True)
        self.attn_mask_2 = AttentionMask()
        
        # Avg pooling
        self.avg_pooling_1 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_2 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_3 = nn.AvgPool2d(self.pool_size)
        
        # Dropout layers
        self.dropout_1 = nn.Dropout(self.dropout_rate1)
        self.dropout_2 = nn.Dropout(self.dropout_rate1)
        self.dropout_3 = nn.Dropout(self.dropout_rate1)
        self.dropout_4 = nn.Dropout(self.dropout_rate2)
        
        # Dense layers
        if img_size == 36:
            self.final_dense_1 = nn.Linear(3136, self.nb_dense, bias=True)
        elif img_size == 72:
            self.final_dense_1 = nn.Linear(16384, self.nb_dense, bias=True)
        elif img_size == 96:
            self.final_dense_1 = nn.Linear(30976, self.nb_dense, bias=True)
        else:
            raise Exception(f'Unsupported image size: {img_size}')
        
        self.final_dense_2 = nn.Linear(self.nb_dense, 1, bias=True)
    
    def forward(self, inputs, params=None):
        """
        前向传播
        
        Args:
            inputs: 输入tensor, shape (batch, 6, H, W)
                   前3通道为差分归一化，后3通道为标准化
            
        Returns:
            输出预测值
        """
        diff_input = inputs[:, :3, :, :]  # 差分部分
        raw_input = inputs[:, 3:, :, :]   # 原始部分
        
        # Motion branch
        d1 = torch.tanh(self.motion_conv1(diff_input))
        d2 = torch.tanh(self.motion_conv2(d1))
        
        # Appearance branch
        r1 = torch.tanh(self.appearance_conv1(raw_input))
        r2 = torch.tanh(self.appearance_conv2(r1))
        
        # First attention
        g1 = torch.sigmoid(self.appearance_att_conv1(r2))
        g1 = self.attn_mask_1(g1)
        gated1 = d2 * g1
        
        d3 = self.avg_pooling_1(gated1)
        d4 = self.dropout_1(d3)
        
        r3 = self.avg_pooling_2(r2)
        r4 = self.dropout_2(r3)
        
        # Second stage
        d5 = torch.tanh(self.motion_conv3(d4))
        d6 = torch.tanh(self.motion_conv4(d5))
        
        r5 = torch.tanh(self.appearance_conv3(r4))
        r6 = torch.tanh(self.appearance_conv4(r5))
        
        # Second attention
        g2 = torch.sigmoid(self.appearance_att_conv2(r6))
        g2 = self.attn_mask_2(g2)
        gated2 = d6 * g2
        
        d7 = self.avg_pooling_3(gated2)
        d8 = self.dropout_3(d7)
        
        # Flatten and dense layers
        d9 = d8.view(d8.size(0), -1)
        d10 = torch.tanh(self.final_dense_1(d9))
        d11 = self.dropout_4(d10)
        out = self.final_dense_2(d11)
        
        return out


class DeepPhysModel:
    """DeepPhys模型封装类"""
    
    def __init__(self, model_path, device='cpu'):
        """
        初始化模型
        
        Args:
            model_path: 模型权重文件路径
            device: 运行设备 (cpu/cuda)
        """
        self.device = torch.device(device)
        
        print(f"正在加载模型: {model_path}")
        
        # 创建模型
        self.model = DeepPhys(
            in_channels=3,
            nb_filters1=32,
            nb_filters2=64,
            kernel_size=3,
            dropout_rate1=0.25,
            dropout_rate2=0.5,
            pool_size=(2, 2),
            nb_dense=128,
            img_size=72
        )
        
        # 加载权重
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
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
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"模型加载成功! 设备: {self.device}")
    
    @torch.no_grad()
    def predict(self, input_tensor):
        """
        单次预测
        
        Args:
            input_tensor: 输入tensor, shape (1, 6, 72, 72)
            
        Returns:
            预测结果 tensor
        """
        input_tensor = input_tensor.to(self.device)
        output = self.model(input_tensor)
        return output
    
    @torch.no_grad()
    def predict_batch(self, input_batch):
        """
        批量预测
        
        Args:
            input_batch: 输入batch, shape (N, 6, 72, 72)
            
        Returns:
            预测结果 tensor
        """
        input_batch = input_batch.to(self.device)
        output = self.model(input_batch)
        return output
