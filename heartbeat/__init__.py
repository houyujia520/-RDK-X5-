"""
实时心率检测统一接口模块
整合人脸检测和rPPG算法，提供简洁的API供main.py调用
"""

import cv2
import numpy as np
from collections import deque
from .face_detector import FaceDetector
from .rppg_algorithm import RPPGProcessor
from .config import SAMPLING_FREQUENCY, MIN_SIGNAL_SIZE, MAX_SIGNAL_SIZE, RESCAN_FREQUENCY


class RealTimeRPPG:
    """
    实时心率检测器
    封装了人脸检测、ROI提取和rPPG信号处理的完整流程
    """
    
    def __init__(self, opt=None):
        """
        初始化心率检测器
        
        Args:
            opt: 配置对象（可选），包含以下属性：
                - heartbeat_model: Haar级联分类器路径
                - hr_min: 最小心率值
                - hr_max: 最大心率值
                - window_size: 分析窗口大小
        """
        # 加载人脸检测器
        model_path = getattr(opt, 'heartbeat_model', None) if opt else None
        self.face_detector = FaceDetector(method='haar')
        
        # 初始化rPPG处理器
        fs = SAMPLING_FREQUENCY
        min_signal = MIN_SIGNAL_SIZE
        max_signal = MAX_SIGNAL_SIZE
        
        if opt:
            if hasattr(opt, 'window_size'):
                max_signal = opt.window_size
                min_signal = min(100, max_signal // 3)
        
        self.rppg_processor = RPPGProcessor(
            fs=fs,
            min_signal_size=min_signal,
            max_signal_size=max_signal
        )
        
        # 状态变量
        self.current_face_box = None
        self.frame_counter = 0
        self.sqi_history = deque(maxlen=30)  # 记录最近30帧的信号质量
        
        print(f"[Heartbeat] 初始化完成 | FPS={fs}, Window=[{min_signal}, {max_signal}]")
    
    def process_frame(self, frame):
        """
        处理单帧图像，返回心率数据
        
        Args:
            frame: BGR格式的图像帧
            
        Returns:
            tuple: (bpm, sqi, avg3s_hr)
                - bpm: 当前心率值（BPM），无效时为0
                - sqi: 信号质量指标（0-1）
                - avg3s_hr: 最近3秒平均心率
        """
        self.frame_counter += 1
        
        # 转换为灰度图用于人脸检测
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 定期重新检测人脸
        if self.current_face_box is None or self.frame_counter % RESCAN_FREQUENCY == 0:
            self.current_face_box = self.face_detector.detect(gray)
            
            if self.current_face_box is not None:
                x, y, w, h = self.current_face_box
                print(f"[Heartbeat] 检测到人脸: ({x},{y},{w},{h})")
        
        # 如果没有检测到人脸，返回无效数据
        if self.current_face_box is None:
            return 0.0, 0.0, 0.0
        
        # 提取ROI区域（额头）
        roi = self.face_detector.get_roi(frame, self.current_face_box)
        
        if roi.size == 0:
            return 0.0, 0.0, 0.0
        
        # 提取原始RGB信号
        raw_rgb = self.rppg_processor.extract_raw_signal(roi)
        
        # 处理信号并计算心率
        bpm, valid = self.rppg_processor.process_frame(raw_rgb)
        
        # 计算信号质量指标（SQI）
        sqi = self._calculate_sqi(valid, roi)
        
        # 计算最近3秒平均心率
        avg3s_hr = self._calculate_avg3s_hr(bpm, valid)
        
        return float(bpm), float(sqi), float(avg3s_hr)
    
    def _calculate_sqi(self, valid, roi):
        """
        计算信号质量指标（Signal Quality Index）
        
        Args:
            valid: rPPG处理是否有效
            roi: ROI区域图像
            
        Returns:
            float: SQI值（0-1）
        """
        if not valid:
            sqi = 0.0
        else:
            # 基于ROI亮度和对比度计算基础质量
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
            brightness = np.mean(roi_gray)
            contrast = np.std(roi_gray)
            
            # 亮度评分（理想范围50-200）
            brightness_score = 1.0 if 50 <= brightness <= 200 else max(0, 1 - abs(brightness - 125) / 125)
            
            # 对比度评分（越高越好，表示有足够的纹理）
            contrast_score = min(1.0, contrast / 30.0)
            
            sqi = (brightness_score + contrast_score) / 2.0
        
        self.sqi_history.append(sqi)
        return sqi
    
    def _calculate_avg3s_hr(self, current_bpm, valid):
        """
        计算最近3秒的平均心率
        
        Args:
            current_bpm: 当前心率值
            valid: 当前数据是否有效
            
        Returns:
            float: 平均心率值
        """
        if not hasattr(self, '_bpm_buffer'):
            self._bpm_buffer = deque(maxlen=90)  # 3秒@30fps = 90帧
            self._timestamp_buffer = deque(maxlen=90)
        
        import time
        current_time = time.time()
        
        if valid and current_bpm > 0:
            self._bpm_buffer.append(current_bpm)
            self._timestamp_buffer.append(current_time)
        
        # 清理超过3秒的数据
        while self._timestamp_buffer and (current_time - self._timestamp_buffer[0]) > 3.0:
            self._bpm_buffer.popleft()
            self._timestamp_buffer.popleft()
        
        # 计算平均值
        if self._bpm_buffer:
            return sum(self._bpm_buffer) / len(self._bpm_buffer)
        else:
            return 0.0
    
    def reset(self):
        """重置检测器状态"""
        self.current_face_box = None
        self.frame_counter = 0
        self.sqi_history.clear()
        if hasattr(self, '_bpm_buffer'):
            self._bpm_buffer.clear()
            self._timestamp_buffer.clear()
        print("[Heartbeat] 检测器已重置")
