"""人脸检测模块 - 使用Haar Cascade"""

import cv2
import numpy as np
import os


class FaceDetector:
    """基于Haar Cascade的人脸检测器"""
    
    def __init__(self, use_large_box=True, large_box_coef=1.5):
        """
        初始化人脸检测器
        
        Args:
            use_large_box: 是否扩大检测框
            large_box_coef: 扩大系数
        """
        # 加载Haar Cascade分类器
        cascade_path = self._find_cascade_file()
        
        if cascade_path is None:
            raise FileNotFoundError(
                "无法找到haarcascade_frontalface_default.xml文件\n"
            )
        
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.use_large_box = use_large_box
        self.large_box_coef = large_box_coef
        
        print(f"人脸检测器已初始化 (use_large_box={use_large_box}, coef={large_box_coef})")
    
    def _find_cascade_file(self):
        """查找Haar Cascade文件"""
        # 可能的路径列表
        possible_paths = [
            # 当前目录
            'haarcascade_frontalface_default.xml',
            # utils目录
            os.path.join('utils', 'haarcascade_frontalface_default.xml'),
            # 从rPPG-Toolbox复制的路径
            os.path.join('..', 'rPPG-Toolbox-main', 'dataset', 
                        'haarcascade_frontalface_default.xml'),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def detect(self, frame):
        """
        检测人脸
        
        Args:
            frame: BGR格式的图像帧
            
        Returns:
            (x, y, w, h) 人脸边界框，如果未检测到则返回None
        """
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 直方图均衡化以提高检测效果
        gray = cv2.equalizeHist(gray)
        
        # 检测人脸
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) == 0:
            return None
        
        # 选择最大的人脸（假设最靠近摄像头的人脸最重要）
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        
        x, y, w, h = largest_face
        
        # 如果需要扩大检测框
        if self.use_large_box:
            center_x = x + w // 2
            center_y = y + h // 2
            
            new_w = int(w * self.large_box_coef)
            new_h = int(h * self.large_box_coef)
            
            x = max(0, center_x - new_w // 2)
            y = max(0, center_y - new_h // 2)
            w = min(new_w, frame.shape[1] - x)
            h = min(new_h, frame.shape[0] - y)
        
        return (int(x), int(y), int(w), int(h))
    
    def detect_multiple(self, frame):
        """
        检测多个人脸
        
        Args:
            frame: BGR格式的图像帧
            
        Returns:
            人脸边界框列表 [(x, y, w, h), ...]
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if self.use_large_box and len(faces) > 0:
            expanded_faces = []
            for (x, y, w, h) in faces:
                center_x = x + w // 2
                center_y = y + h // 2
                
                new_w = int(w * self.large_box_coef)
                new_h = int(h * self.large_box_coef)
                
                new_x = max(0, center_x - new_w // 2)
                new_y = max(0, center_y - new_h // 2)
                new_w = min(new_w, frame.shape[1] - new_x)
                new_h = min(new_h, frame.shape[0] - new_y)
                
                expanded_faces.append((new_x, new_y, new_w, new_h))
            
            return expanded_faces
        
        return [tuple(face) for face in faces]
