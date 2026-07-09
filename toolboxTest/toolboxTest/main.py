"""
rPPG实时心率检测系统 - Flask Web服务版本 (RDK摄像头)
使用DeepPhys模型从RDK摄像头视频中实时检测心率,通过Web界面展示
"""

import argparse
import cv2
import numpy as np
import torch
import time
import threading
from collections import deque
from flask import Flask, Response, jsonify
from hobot_vio import libsrcampy
from utils.face_detector import FaceDetector
from utils.signal_processor import SignalProcessor
from models.deepphys_model import DeepPhysModel


# ============================================================
# Flask App Setup
# ============================================================
app = Flask(__name__)

@app.after_request
def after_request(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


# ============================================================
# Global Shared Resources
# ============================================================
latest_frame = None
latest_heart_rate = {
    "bpm": 0.0,
    "sqi": 0.0,
    "avg3s_hr": 0.0
}

lock = threading.Lock()
stop_event = threading.Event()


# ============================================================
# RDK Camera Class (Hardware Accelerated)
# ============================================================
class RDKCamera:
    """RDK X5/X3 硬件加速摄像头"""
    def __init__(self, camera_id=0, width=640, height=480, fps=30):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps

        self.cam = libsrcampy.Camera()
        ret = self.cam.open_cam(camera_id, -1, fps, width, height)
        if ret != 0:
            raise RuntimeError(f"Failed to open RDK camera (error code: {ret})")

        print(f"[INFO] RDK摄像头已打开: {width}x{height}@{fps}fps")
        time.sleep(1)  # 等待摄像头稳定

    def read(self):
        """读取一帧图像"""
        img_data = self.cam.get_img(2)
        if img_data is None:
            return False, None

        # YUV NV21格式转换为BGR
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        
        # 根据实际数据大小计算真实分辨率
        # YUV NV21格式: Y平面(width*height) + UV平面(width*height/2) = width*height*1.5
        total_pixels = len(img_array)
        
        # 尝试推断实际分辨率
        # 常见分辨率: 1920x1080, 1280x720, 640x480等
        if total_pixels == 1920 * 1080 * 3 // 2:
            actual_height = 1080
            actual_width = 1920
        elif total_pixels == 1280 * 720 * 3 // 2:
            actual_height = 720
            actual_width = 1280
        elif total_pixels == 640 * 480 * 3 // 2:
            actual_height = 480
            actual_width = 640
        else:
            # 通用计算: height = sqrt(total_pixels / 1.5 / aspect_ratio)
            # 假设16:9或4:3宽高比
            aspect_ratio = 16/9
            actual_height = int(np.sqrt(total_pixels / 1.5 / aspect_ratio))
            actual_width = int(actual_height * aspect_ratio)
            
            # 验证计算是否正确
            if actual_width * actual_height * 3 // 2 != total_pixels:
                print(f"[WARN] 无法解析YUV数据大小: {total_pixels}")
                print(f"[WARN] 期望大小: {self.width}x{self.height}x1.5 = {self.width * self.height * 3 // 2}")
                return False, None
        
        # 使用实际分辨率重塑数组
        yuv = img_array.reshape((actual_height + actual_height // 2, actual_width))
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)
        
        # 如果实际分辨率与期望不同,进行缩放
        if actual_width != self.width or actual_height != self.height:
            bgr = cv2.resize(bgr, (self.width, self.height))
            if actual_width != self.width or actual_height != self.height:
                print(f"[INFO] 摄像头实际分辨率: {actual_width}x{actual_height}, 已缩放到: {self.width}x{self.height}")
        
        return True, bgr

    def release(self):
        """释放摄像头资源"""
        self.cam.close_cam()
        print("[INFO] RDK摄像头已关闭")


# ============================================================
# Heart Rate Calculator
# ============================================================
class HeartRateCalculator:
    """
    高级心率计算器
    包含自适应滤波、谐波抑制、抛物线插值和 SQI 质量门控
    """
    def __init__(self, fps=30, min_hr=40, max_hr=220):
        self.fps = fps
        self.min_hr = min_hr
        self.max_hr = max_hr
        
        # 信号缓冲 (使用原始模型输出或预处理后的信号)
        self.signal_buffer = deque(maxlen=int(fps * 10))  # 10秒窗口
        
        # 状态跟踪
        self.last_valid_hr = 75.0  # 初始假设心率
        self.consecutive_low_sqi = 0
        
        # 平滑参数
        self.alpha = 0.15  # EWMA 平滑系数
        
    def add_signal(self, value):
        """添加新的信号值"""
        self.signal_buffer.append(value)
        
    def calculate_hr(self):
        """
        计算当前心率
        返回: (hr, sqi) -> 心率, 信号质量指数(0-1)
        """
        if len(self.signal_buffer) < int(self.fps * 4):  # 至少需要4秒数据
            return self.last_valid_hr, 0.0
            
        signal = np.array(self.signal_buffer)
        
        # 1. 预处理：去趋势和标准化
        processed_signal = self._preprocess(signal)
        
        # 2. 动态带通滤波 (关键：聚焦于上一帧心率附近)
        filtered_signal = self._adaptive_filter(processed_signal)
        
        # 3. 频域分析 (Welch PSD 比直接 FFT 更稳健)
        from scipy.signal import welch
        freqs, psd = welch(filtered_signal, fs=self.fps, nperseg=min(len(filtered_signal), 256))
        
        # 4. 限制感兴趣频段
        min_freq = self.min_hr / 60.0
        max_freq = self.max_hr / 60.0
        mask = (freqs >= min_freq) & (freqs <= max_freq)
        valid_freqs = freqs[mask]
        valid_psd = psd[mask]
        
        if len(valid_psd) == 0:
            return self.last_valid_hr, 0.0
            
        # 1. 寻找主频峰值
        peak_idx = np.argmax(valid_psd)
        peak_freq = valid_freqs[peak_idx]
        peak_power = valid_psd[peak_idx]
        
        # 2. 计算基础 SNR
        noise_power = np.mean(valid_psd)
        snr = peak_power / (noise_power + 1e-6)
            
        # 3. 【新增】计算频谱平坦度 (Spectral Flatness)
        # 几何均值 / 算术均值。越接近 0 表示峰值越尖锐（信号越好），越接近 1 表示越像噪声
        eps = 1e-10
        log_psd = np.log(valid_psd + eps)
        geometric_mean = np.exp(np.mean(log_psd))
        arithmetic_mean = np.mean(valid_psd + eps)
        spectral_flatness = geometric_mean / arithmetic_mean
        
        # 4. 抛物线插值
        refined_freq = self._parabolic_interpolation(valid_freqs, valid_psd, peak_idx)
        if refined_freq is not None:
            peak_freq = refined_freq
            
        calculated_hr = peak_freq * 60.0
        
        # 5. 谐波检查
        corrected_hr = self._check_harmonics(calculated_hr, self.last_valid_hr)
        
        # 6. 【优化】综合 SQI 计算
        # 结合 SNR 和频谱平坦度。如果频谱太平坦（噪声），即使 SNR 高也要降低 SQI
        snr_score = min(1.0, snr / 4.0)
        flatness_score = max(0, 1.0 - spectral_flatness * 2.0) # 平坦度越低分越高
        combined_sqi = snr_score * 0.7 + flatness_score * 0.3
        
        # 7. 【关键】心率变化率约束
        hr_change = abs(corrected_hr - self.last_valid_hr)
        max_allowed_change = 10.0  # 每次计算允许的最大合理变化
        
        # 如果变化过大且 SQI 不是极高，则认为是噪声跳变
        if hr_change > max_allowed_change and combined_sqi < 0.8:
            combined_sqi *= 0.5  # 惩罚性降低 SQI
            
        # 8. 最终决策
        if combined_sqi > 0.3:
            self.consecutive_low_sqi = 0
            
            # 动态平滑：如果心率很高且在下降，或者变化剧烈，使用更强的平滑
            if corrected_hr > 150 or hr_change > 15:
                dynamic_alpha = self.alpha * 0.5
            else:
                dynamic_alpha = self.alpha
                
            final_hr = self.last_valid_hr * (1 - dynamic_alpha) + corrected_hr * dynamic_alpha
            self.last_valid_hr = final_hr
        else:
            self.consecutive_low_sqi += 1
            # 【新增】高位衰减：如果心率在高位且信号不好，加速回归静息心率
            if self.last_valid_hr > 120:
                decay_rate = 0.95  # 快速衰减
            else:
                decay_rate = 0.99  # 缓慢衰减
                
            final_hr = self.last_valid_hr * decay_rate + 70 * (1 - decay_rate)
                
        return final_hr, combined_sqi

    def _preprocess(self, signal):
        """去趋势和标准化"""
        detrended = signal - np.linspace(signal[0], signal[-1], len(signal))
        if np.std(detrended) == 0:
            return detrended
        return (detrended - np.mean(detrended)) / np.std(detrended)

    def _adaptive_filter(self, signal):
        """自适应带通滤波"""
        low_cut = self.min_hr / 60.0
        high_cut = self.max_hr / 60.0
        
        # 动态范围：中心频率 +/- 0.33Hz (约 +/- 20bpm)
        if self.last_valid_hr > 0:
            center_freq = self.last_valid_hr / 60.0
            dynamic_low = max(low_cut, center_freq - 0.33)
            dynamic_high = min(high_cut, center_freq + 0.33)
            
            if dynamic_high - dynamic_low < 0.3:
                dynamic_low = center_freq - 0.15
                dynamic_high = center_freq + 0.15
                
            low_cut = dynamic_low
            high_cut = dynamic_high

        nyq = 0.5 * self.fps
        low = low_cut / nyq
        high = high_cut / nyq
        
        if low <= 0 or high >= 1 or low >= high:
            return signal
            
        from scipy.signal import butter, filtfilt
        b, a = butter(4, [low, high], btype='band')
        try:
            filtered = filtfilt(b, a, signal)
        except:
            filtered = signal
        return filtered

    def _parabolic_interpolation(self, freqs, psd, peak_idx):
        """抛物线插值细化峰值"""
        if peak_idx <= 0 or peak_idx >= len(psd) - 1:
            return None
        
        x0, x1, x2 = freqs[peak_idx-1], freqs[peak_idx], freqs[peak_idx+1]
        y0, y1, y2 = psd[peak_idx-1], psd[peak_idx], psd[peak_idx+1]
        
        denom = (y0 - 2*y1 + y2)
        if denom == 0:
            return x1
            
        delta = 0.5 * (y0 - y2) / denom
        refined_x = x1 + delta * (x1 - x0)
        
        if abs(delta) < 1: 
            return refined_x
        return None

    def _check_harmonics(self, hr, last_hr):
        """
        谐波检查：防止 1/2, 1/3, 2x 错误
        并执行强制阈值校正
        """
        if hr <= 0:
            return self.last_valid_hr
            
        # 1. 【新增】强制阈值校正机制
        # 如果心率低于 40，极大概率是真实心率的 1/2（如 75 -> 37.5）
        if hr < 41:
            corrected = hr * 2.0
            print(f"[校正] 检测到极低心率 {hr:.1f}，尝试加倍至 {corrected:.1f}")
            return corrected
            
        # 如果心率高于 140，极大概率是真实心率的 2 倍（如 75 -> 150）或运动噪声
        elif hr > 140:
            corrected = hr / 2.0
            print(f"[校正] 检测到极高高心率 {hr:.1f}，尝试减半至 {corrected:.1f}")
            return corrected

        # 2. 基于上一帧的动态谐波检查（保留原有逻辑作为补充）
        if last_hr > 0:
            r = hr / last_hr
            
            # 检查是否是上一帧的倍数或分数
            if 0.45 < r < 0.55: # 半频错误
                return hr * 2.0
            if 1.8 < r < 2.2:  # 倍频错误
                return hr / 2.0
                    
        return hr


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='rPPG实时心率检测 - RDK Web服务版')
    parser.add_argument('--model_path', type=str, 
                       default='models/SCAMPS_DeepPhys.pth',
                       help='预训练模型路径')
    parser.add_argument('--camera_id', type=int, default=0,
                       help='摄像头ID (默认0)')
    parser.add_argument('--fps', type=int, default=30,
                       help='目标帧率 (默认30)')
    parser.add_argument('--window_size', type=int, default=300,
                       help='分析窗口大小,单位:帧数 (默认300帧=10秒@30fps)')
    parser.add_argument('--use_gpu', action='store_true', default=False,
                       help='是否使用GPU加速')
    
    # RDK摄像头选项
    parser.add_argument('--use_rdk', action='store_true', default=True,
                       help='使用RDK硬件加速摄像头 (默认启用)')
    
    # 心率平滑参数
    parser.add_argument('--hr_min', type=float, default=40.0,
                       help='最小合理心率 (默认40 BPM)')
    parser.add_argument('--hr_max', type=float, default=220.0,
                       help='最大合理心率 (默认220 BPM)')
    parser.add_argument('--ewma_alpha', type=float, default=0.15,
                       help='指数加权移动平均系数 (0-1, 越小越平滑, 默认0.15)')
    parser.add_argument('--hr_update_interval', type=int, default=15,
                       help='心率更新间隔帧数 (默认每15帧更新一次)')
    parser.add_argument('--median_filter_size', type=int, default=7,
                       help='中值滤波窗口大小 (默认7, 设为1禁用)')
    
    # Flask服务器参数
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='服务器监听地址 (默认0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000,
                       help='服务器端口 (默认5000)')
    
    return parser.parse_args()


class RealTimeRPPG:
    """实时rPPG检测器"""
    
    def __init__(self, args):
        # 3 秒平均心率相关
        self.avg3s_hr = 0.0
        self.avg3s_buffer = deque(maxlen=6)  # 存 6 个 1 秒心率
        self.avg3s_timer = time.time()
        
        
        self.args = args
        self.device = torch.device('cuda:0' if args.use_gpu and torch.cuda.is_available() else 'cpu')
        
        print(f"使用设备: {self.device}")
        
        # 初始化组件
        self.face_detector = FaceDetector()
        self.signal_processor = SignalProcessor(fs=args.fps)
        self.model = DeepPhysModel(args.model_path, self.device)
        
        # 初始化高级心率计算器
        self.hr_calculator = HeartRateCalculator(
            fps=args.fps,
            min_hr=args.hr_min,
            max_hr=args.hr_max
        )
        
        # 帧缓冲区
        self.frame_buffer = deque(maxlen=args.window_size)
        self.face_buffer = deque(maxlen=args.window_size)
        
        # 心率结果
        self.current_hr = 0.0
        self.raw_hr = 0.0
        self.sqi = 0.0
        self.avg3s_hr = 0.0
        self.avg3s_buffer = deque(maxlen=6)
        self.avg3s_timer = time.time()
        
        # 性能统计
        self.fps_counter = deque(maxlen=30)
        self.last_fps_time = time.time()
        
        # 更新计时器
        self.last_hr_update_time = time.time()
        self.hr_update_count = 0
        
    def preprocess_frame(self, frame, face_box):
        """
        预处理单帧图像
        返回: 预处理后的6通道tensor (3通道差分归一化 + 3通道标准化)
        """
        x, y, w, h = face_box
        
        # 裁剪人脸区域
        face_img = frame[y:y+h, x:x+w]
        
        # 调整到72x72
        face_resized = cv2.resize(face_img, (72, 72))
        
        # 转换为float32并归一化到[0, 1]
        face_normalized = face_resized.astype(np.float32) / 255.0
        
        return face_normalized
    
    def create_model_input(self):
        """
        从帧缓冲区创建模型输入
        返回: shape为(1, 6, 72, 72)的tensor
        """
        if len(self.face_buffer) < 2:
            return None
        
        # 获取最近的帧
        frames = list(self.face_buffer)
        
        # 计算差分帧 (当前帧 - 前一帧)
        diff_frames = []
        for i in range(1, len(frames)):
            diff = frames[i] - frames[i-1]
            diff_frames.append(diff)
        
        if len(diff_frames) == 0:
            return None
        
        # 取最后一帧作为代表
        last_diff = diff_frames[-1]
        last_raw = frames[-1]
        
        # DiffNormalization: 差分归一化
        diff_mean = np.mean(last_diff, axis=(0, 1), keepdims=True)
        diff_std = np.std(last_diff, axis=(0, 1), keepdims=True) + 1e-8
        diff_normalized = (last_diff - diff_mean) / diff_std
        
        # Standardization: 标准化
        raw_mean = np.mean(last_raw, axis=(0, 1), keepdims=True)
        raw_std = np.std(last_raw, axis=(0, 1), keepdims=True) + 1e-8
        standardized = (last_raw - raw_mean) / raw_std
        
        # 拼接: [diff_normalized, standardized] -> shape (72, 72, 6)
        combined = np.concatenate([diff_normalized, standardized], axis=2)
        
        # 转换为tensor: (H, W, C) -> (C, H, W) -> (1, C, H, W)
        tensor_input = torch.from_numpy(combined.transpose(2, 0, 1)).unsqueeze(0).float()
        
        return tensor_input
    
    def process_frame(self, frame):
        """处理单帧图像(不显示,只处理数据)"""
        global latest_frame, latest_heart_rate
        
        # 检测人脸
        face_box = self.face_detector.detect(frame)
        
        if face_box is not None:
            # 预处理帧
            processed = self.preprocess_frame(frame, face_box)
            self.face_buffer.append(processed)
        
        # 如果缓冲区足够,进行推理
        if len(self.face_buffer) >= 30:  # 至少1秒数据
            model_input = self.create_model_input()
            
            if model_input is not None:
                # 模型推理
                prediction = self.model.predict(model_input)
                
                # 将模型输出添加到计算器
                raw_signal = prediction.item() if isinstance(prediction, torch.Tensor) else prediction
                self.hr_calculator.add_signal(raw_signal)
                
                # 计算心率
                if len(self.face_buffer) >= 60:  # 增加到60帧以提高稳定性
                    # 获取计算结果
                    final_hr, sqi = self.hr_calculator.calculate_hr()
                    
                    self.raw_hr = final_hr
                    self.sqi = sqi
                    
                    # 更新显示的心率
                    if sqi > 0.2:  # 只有当有一定信号质量时才更新显示
                        self.current_hr = final_hr
                        self.avg3s_buffer.append(final_hr)
                    
                    # 定期打印统计信息
                    current_time = time.time()
                    if current_time - self.last_hr_update_time > 5.0:
                        print(f"[统计] HR: {self.current_hr:.1f} | "
                              f"SQI: {sqi:.2f} | "
                              f"更新次数: {self.hr_update_count}")
                        self.last_hr_update_time = current_time
                        self.hr_update_count += 1
        
        # 每秒更新一次 3 秒平均心率
        now = time.time()
        if now - self.avg3s_timer >= 2.0:
            self.avg3s_timer = now
            if len(self.avg3s_buffer) > 0:
                self.avg3s_hr = float(np.mean(list(self.avg3s_buffer)))
        
        # 更新全局共享数据
        with lock:
            latest_frame = frame.copy()
            latest_heart_rate["bpm"] = self.current_hr
            latest_heart_rate["sqi"] = self.sqi
            latest_heart_rate["avg3s_hr"] = self.avg3s_hr
        
        return frame
    
    def run_detection(self):
        """运行实时检测(后台线程)"""
        print("正在打开RDK摄像头...")
        
        camera = RDKCamera(
            camera_id=self.args.camera_id,
            width=640,
            height=480,
            fps=self.args.fps
        )
        
        print("="*60)
        print("开始实时检测...")
        print(f"心率范围: {self.args.hr_min}-{self.args.hr_max} BPM")
        print(f"EWMA系数: {self.args.ewma_alpha}")
        print("按 Ctrl+C 停止服务")
        print("="*60)
        
        try:
            while not stop_event.is_set():
                ret, frame = camera.read()
                
                if not ret or frame is None:
                    print("错误: 无法读取帧")
                    break
                
                # 处理帧
                self.process_frame(frame)
                    
        finally:
            camera.release()
            
            print("\n" + "="*60)
            print("检测结束 - 最终统计")
            print(f"最终心率: {self.current_hr:.1f} BPM")
            print(f"最终 SQI: {self.sqi:.2f}")
            print("="*60)


# ============================================================
# Flask Routes
# ============================================================

# 全局检测器实例
detector = None

def generate_video_stream():
    """生成MJPEG视频流"""
    global latest_frame
    
    while True:
        frame = None
        with lock:
            if latest_frame is not None:
                frame = latest_frame.copy()
        
        if frame is not None:
            # 缩小帧以降低带宽
            frame_small = cv2.resize(frame, (640, 360))
            
            # 绘制心率信息到视频帧上
            if detector and detector.current_hr > 0:
                hr_text = f"HR: {detector.current_hr:.1f} BPM"
                cv2.putText(frame_small, hr_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 编码为JPEG
            ret, jpeg = cv2.imencode(".jpg", frame_small, [cv2.IMWRITE_JPEG_QUALITY, 70])
            
            if ret:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" +
                    jpeg.tobytes() +
                    b"\r\n"
                )
        
        # 控制帧率
        time.sleep(0.033)  # 约30fps


@app.route("/video_feed")
def video_feed():
    """视频流端点"""
    return Response(
        generate_video_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/status")
def status():
    """心率状态端点"""
    with lock:
        return jsonify({
            "bpm": latest_heart_rate["bpm"],
            "sqi": latest_heart_rate["sqi"],
            "avg3s_hr": latest_heart_rate["avg3s_hr"]
        })


@app.route("/drug")
def drug():
    """药品识别端点(预留接口,暂返回空数据)"""
    return jsonify({
        "drug_name": None,
        "advice": None,
        "confidence": 0.0
    })


@app.route("/health")
def health_check():
    """健康检查端点"""
    return {"status": "running"}


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='rPPG实时心率检测 - RDK Web服务版')
    parser.add_argument('--model_path', type=str, 
                       default='models/SCAMPS_DeepPhys.pth',
                       help='预训练模型路径')
    parser.add_argument('--camera_id', type=int, default=0,
                       help='摄像头ID (默认0)')
    parser.add_argument('--fps', type=int, default=30,
                       help='目标帧率 (默认30)')
    parser.add_argument('--window_size', type=int, default=300,
                       help='分析窗口大小,单位:帧数 (默认300帧=10秒@30fps)')
    parser.add_argument('--use_gpu', action='store_true', default=False,
                       help='是否使用GPU加速')
    
    # 心率平滑参数
    parser.add_argument('--hr_min', type=float, default=40.0,
                       help='最小合理心率 (默认40 BPM)')
    parser.add_argument('--hr_max', type=float, default=220.0,
                       help='最大合理心率 (默认220 BPM)')
    parser.add_argument('--ewma_alpha', type=float, default=0.15,
                       help='指数加权移动平均系数 (0-1, 越小越平滑, 默认0.15)')
    parser.add_argument('--hr_update_interval', type=int, default=15,
                       help='心率更新间隔帧数 (默认每15帧更新一次)')
    parser.add_argument('--median_filter_size', type=int, default=7,
                       help='中值滤波窗口大小 (默认7, 设为1禁用)')
    
    # Flask服务器参数
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='服务器监听地址 (默认0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000,
                       help='服务器端口 (默认5000)')
    
    return parser.parse_args()


def main():
    global detector
    
    args = parse_args()
    
    # 检查模型文件是否存在
    import os
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在: {args.model_path}")
        print("请确保模型文件已放置在正确位置")
        return
    
    # 创建检测器
    detector = RealTimeRPPG(args)
    
    # 启动检测线程
    detection_thread = threading.Thread(target=detector.run_detection, daemon=True)
    detection_thread.start()
    
    # 启动Flask服务器
    print(f"\n[INFO] Web服务启动中...")
    print(f"[INFO] 视频流地址: http://{args.host}:{args.port}/video_feed")
    print(f"[INFO] 心率数据: http://{args.host}:{args.port}/status")
    print("="*60)
    
    try:
        app.run(host=args.host, port=args.port, threaded=True)
    except KeyboardInterrupt:
        print("\n服务已停止")
    finally:
        stop_event.set()


if __name__ == '__main__':
    main()
