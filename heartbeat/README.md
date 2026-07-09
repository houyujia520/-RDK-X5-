# 心率检测模块 (heartbeat)

## 概述

本模块实现了基于 rPPG（远程光电容积脉搏波）技术的实时心率检测功能，通过普通摄像头即可非接触式测量心率。

## 文件结构

```
heartbeat/
├── __init__.py              # 主接口模块，提供 RealTimeRPPG 类
├── config.py                # 配置文件（采样率、窗口大小等参数）
├── face_detector.py         # 人脸检测模块（Haar级联分类器）
├── rppg_algorithm.py        # rPPG算法核心实现（POS算法）
└── models/
    └── haarcascade_frontalface_alt.xml  # Haar级联分类器模型
```

## 核心组件

### 1. RealTimeRPPG 类（`__init__.py`）

统一的对外接口，封装了完整的心率检测流程：

```python
from heartbeat import RealTimeRPPG

# 初始化
detector = RealTimeRPPG(opt)

# 处理帧
bpm, sqi, avg3s = detector.process_frame(frame)
```

**返回值说明：**
- `bpm`: 当前心率值（BPM），无效时为 0
- `sqi`: 信号质量指标（0-1），越高表示信号越可靠
- `avg3s_hr`: 最近3秒的平均心率

### 2. FaceDetector 类（`face_detector.py`）

负责检测人脸并提取 ROI（感兴趣区域）：
- 使用 Haar 级联分类器检测人脸
- 自动提取额头区域作为 ROI（避免眼睛、嘴巴干扰）
- 支持定期重新检测（每10帧）

### 3. RPPGProcessor 类（`rppg_algorithm.py`）

核心算法实现，包含以下步骤：
1. **信号提取**: 从 ROI 提取平均 RGB 值
2. **去趋势处理**: 使用移动平均消除光照变化
3. **POS 算法**: 投影到颜色空间分离血流信号
4. **带通滤波**: 滤除 42-210 BPM 范围外的噪声
5. **频域分析**: FFT 计算心率主频
6. **时域验证**: 峰值检测交叉验证
7. **融合策略**: 结合频域和时域结果提高准确性

## 配置参数（`config.py`）

```python
SAMPLING_FREQUENCY = 30   # 采样频率 (Hz)
RESCAN_FREQUENCY = 10     # 人脸重新检测频率（每N帧）
MIN_SIGNAL_SIZE = 100     # 信号处理最小窗口大小
MAX_SIGNAL_SIZE = 300     # 信号处理最大窗口大小（10秒@30fps）
```

## 使用方法

### 在 main.py 中集成

```python
from heartbeat import RealTimeRPPG

# 初始化检测器（懒加载）
heartbeat_detector = None

# 在处理循环中
if mode == 'heartbeat':
    if heartbeat_detector is None:
        heartbeat_detector = RealTimeRPPG(opt)
    
    bpm, sqi, avg3s = heartbeat_detector.process_frame(frame)
```

### 独立测试

```bash
python test_heartbeat.py
```

## 技术细节

### rPPG 原理

rPPG 技术利用普通摄像头捕捉面部皮肤微小的颜色变化（由血液流动引起），通过信号处理算法提取心率信息。

### POS 算法优势

- **抗运动干扰**: 通过颜色空间投影减少头部运动影响
- **抗光照变化**: 移动平均去趋势适应非线性光照变化
- **高准确性**: 频域+时域双重验证

### 信号质量控制

- **亮度检查**: ROI 平均亮度应在 50-200 范围内
- **对比度检查**: 足够的纹理信息确保信号可提取
- **信噪比阈值**: 峰值能量需 >= 2倍噪声基底
- **变异系数**: 心跳间隔变化应 < 15%

## 性能优化建议

1. **光照条件**: 
   - ✅ 使用自然光或直流 LED 灯
   - ❌ 避免荧光灯（50/60Hz 频闪）

2. **拍摄距离**: 
   - 最佳距离 30-50cm
   - 确保额头清晰可见

3. **分辨率**: 
   - 640x480 已足够
   - 更高分辨率会增加处理负担

4. **帧率稳定性**: 
   - 固定 30fps 最佳
   - 避免自动曝光/白平衡

## 常见问题

### Q1: 心率一直显示 0？

**可能原因：**
- 未检测到人脸（检查光线、角度）
- 额头被遮挡（刘海、帽子等）
- 光线不足或过强
- 等待时间不够（需要至少 100 帧约 3.3 秒）

### Q2: 心率数值波动大？

**解决方案：**
- 保持头部静止
- 使用 `avg3s_hr`（3秒平均值）而非瞬时值
- 检查 SQI 指标，只在 SQI > 0.7 时信任数据

### Q3: 模型文件找不到？

确保路径正确：
```python
opt.heartbeat_model = 'heartbeat/models/haarcascade_frontalface_alt.xml'
```

## 参考资料

- [rPPG 技术综述](https://en.wikipedia.org/wiki/Photoplethysmography)
- [POS 算法论文](https://ieeexplore.ieee.org/document/7780928)
- [OpenCV Haar 级联分类器](https://docs.opencv.org/master/d7/d8b/tutorial_py_face_detection.html)

## 许可证

本项目仅供学习和研究使用。
