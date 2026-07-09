import os
import cv2
import numpy as np
from .config import HAAR_CASCADE_PATH

class FaceDetector:
    def __init__(self, method='haar'):
        self.method = method
        if self.method == 'haar':
            if not os.path.exists(HAAR_CASCADE_PATH):
                raise FileNotFoundError(f"Haar cascade file not found at {HAAR_CASCADE_PATH}")
            self.face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
        elif self.method == 'dnn':
            # 如果需要DNN，需加载对应模型，此处暂以Haar为例
            pass

    def detect(self, frame_gray):
        """
        检测人脸并返回边界框 (x, y, w, h)
        """
        faces = self.face_cascade.detectMultiScale(
            frame_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) > 0:
            # 返回面积最大的人脸
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            return largest_face
        return None

    def get_roi(self, frame, face_box):
        """
        根据人脸框获取感兴趣区域 (ROI)，通常选取额头区域
        """
        x, y, w, h = face_box
        # 简单策略：选取上半部分作为额头区域，避免嘴巴眼睛干扰
        roi_h = int(h * 0.5)
        roi_y = y
        roi_w = w
        roi_x = x
        
        # 边界检查
        roi_x = max(0, roi_x)
        roi_y = max(0, roi_y)
        roi_w = min(w, frame.shape[1] - roi_x)
        roi_h = min(roi_h, frame.shape[0] - roi_y)
        
        return frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
