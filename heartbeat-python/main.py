"""
rPPG实时心率检测系统 - Flask Web服务版本 (RDK摄像头)
使用POS算法从RDK摄像头视频中实时检测心率,通过Web界面展示

架构优化: 三线程异步处理
- 线程1: 视频采集(高频,无阻塞)
- 线程2: rPPG算法处理(后台运行)
- 线程3: Web服务(响应请求)
"""

import cv2
import numpy as np
import time
import threading
from collections import deque
from config import *
from face_detector import FaceDetector
from rppg_algorithm import RPPGProcessor

# 尝试导入RDK摄像头库
try:
    from hobot_vio import libsrcampy
    RDK_AVAILABLE = True
except ImportError:
    RDK_AVAILABLE = False
    print("[WARN] hobot_vio库未找到,将使用普通OpenCV摄像头")

import web_stream 


class RDKCamera:
    """RDK X5/X3 硬件加速摄像头"""
    def __init__(self, camera_id=0, width=640, height=480, fps=30):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps

        if not RDK_AVAILABLE:
            raise RuntimeError("RDK摄像头库不可用")

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
        total_pixels = len(img_array)
        
        # 尝试推断实际分辨率
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
            # 通用计算
            aspect_ratio = 16/9
            actual_height = int(np.sqrt(total_pixels / 1.5 / aspect_ratio))
            actual_width = int(actual_height * aspect_ratio)
            
            # 验证计算是否正确
            if actual_width * actual_height * 3 // 2 != total_pixels:
                print(f"[WARN] 无法解析YUV数据大小: {total_pixels}")
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


class VideoCaptureThread:
    """
    视频采集线程 - 高频运行,只负责读取帧并存入缓冲区
    不进行任何耗时处理,确保最高帧率
    """
    def __init__(self, use_rdk=True):
        self.use_rdk = use_rdk
        self.frame_buffer = deque(maxlen=3)  # 只保留最新3帧,避免内存堆积
        self.lock = threading.Lock()
        self.running = False
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.actual_fps = 0
        
        # 初始化摄像头
        if use_rdk and RDK_AVAILABLE:
            print("[INFO] 初始化RDK摄像头...")
            self.cap = RDKCamera(camera_id=0, width=640, height=480, fps=30)
        else:
            print("[INFO] 初始化普通摄像头...")
            self.cap = cv2.VideoCapture(VIDEO_SOURCE)
            if not self.cap.isOpened():
                raise Exception("Could not open video source")
    
    def start(self):
        """启动采集线程"""
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        print("[INFO] 视频采集线程已启动")
    
    def _capture_loop(self):
        """采集循环 - 尽可能快地读取帧"""
        while self.running:
            try:
                if self.use_rdk and RDK_AVAILABLE:
                    ret, frame = self.cap.read()
                else:
                    ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    print("[WARN] 无法读取帧")
                    time.sleep(0.01)
                    continue
                
                # 计算FPS
                self.fps_counter += 1
                current_time = time.time()
                if current_time - self.last_fps_time >= 1.0:
                    self.actual_fps = self.fps_counter / (current_time - self.last_fps_time)
                    self.fps_counter = 0
                    self.last_fps_time = current_time
                    # 不打印,避免影响性能
                
                # 存入缓冲区(线程安全)
                with self.lock:
                    self.frame_buffer.clear()  # 清空旧帧,只保留最新
                    self.frame_buffer.append(frame.copy())
                    
            except Exception as e:
                print(f"[ERROR] 采集错误: {e}")
                time.sleep(0.01)
    
    def get_latest_frame(self):
        """获取最新帧(非阻塞)"""
        with self.lock:
            if self.frame_buffer:
                return self.frame_buffer[-1].copy()
            return None
    
    def stop(self):
        """停止采集"""
        self.running = False
        if self.use_rdk and RDK_AVAILABLE:
            self.cap.release()
        else:
            self.cap.release()
        print("[INFO] 视频采集线程已停止")


class RPPGProcessingThread:
    """
    rPPG算法处理线程 - 后台运行,从视频缓冲区取帧进行处理
    不影响视频采集和推送的帧率
    """
    def __init__(self, video_capture: VideoCaptureThread):
        self.video_capture = video_capture
        self.detector = FaceDetector(method='haar')
        self.processor = RPPGProcessor(fs=30, min_signal_size=MIN_SIGNAL_SIZE, max_signal_size=MAX_SIGNAL_SIZE)
        
        self.running = False
        self.current_bpm = 0
        self.face_box = None
        self.frame_count = 0
        
        # FPS动态更新
        self.last_update_time = time.time()
    
    def start(self):
        """启动处理线程"""
        self.running = True
        self.thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.thread.start()
        print("[INFO] rPPG处理线程已启动")
    
    def _processing_loop(self):
        """处理循环 - 以合理频率运行"""
        while self.running:
            try:
                # 从视频缓冲区获取最新帧
                frame = self.video_capture.get_latest_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                self.frame_count += 1
                
                # 【优化】降低处理频率:每2-3帧处理一次,减轻CPU负担
                if self.frame_count % 2 != 0:
                    time.sleep(0.001)  # 短暂休眠,让出CPU
                    continue
                
                # 动态更新FPS
                current_time = time.time()
                if current_time - self.last_update_time >= 1.0:
                    self.processor.fs = self.video_capture.actual_fps if self.video_capture.actual_fps > 0 else 30
                    self.last_update_time = current_time
                
                # 人脸检测
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if self.face_box is None or (self.frame_count % RESCAN_FREQUENCY == 0):
                    self.face_box = self.detector.detect(gray)
                
                current_bpm = 0
                if self.face_box is not None:
                    roi = self.detector.get_roi(frame, self.face_box)
                    if roi.size > 0:
                        raw_rgb = self.processor.extract_raw_signal(roi)
                        bpm, valid = self.processor.process_frame(raw_rgb)
                        if valid:
                            current_bpm = bpm
                
                # 更新全局心率数据
                if current_bpm > 0:
                    self.current_bpm = current_bpm
                    timestamp = time.time()
                    with web_stream.lock:
                        web_stream.bpm_history.append((timestamp, current_bpm))
                        if len(web_stream.bpm_history) > 100:
                            web_stream.bpm_history.pop(0)
                
                # 短暂休眠,控制处理频率
                time.sleep(0.01)
                
            except Exception as e:
                print(f"[ERROR] 处理错误: {e}")
                time.sleep(0.1)
    
    def stop(self):
        """停止处理"""
        self.running = False
        print("[INFO] rPPG处理线程已停止")


class HeartbeatMonitor:
    """
    主监控类 - 协调视频采集和算法处理
    """
    def __init__(self, use_rdk=True):
        self.use_rdk = use_rdk
        self.running = False
        
        # 创建视频采集线程
        self.video_capture = VideoCaptureThread(use_rdk=use_rdk)
        
        # 创建rPPG处理线程
        self.rppg_processor = RPPGProcessingThread(self.video_capture)
    
    def start(self):
        """启动所有线程"""
        print("="*60)
        print("启动rPPG心率监测系统 (三线程异步架构)")
        print("="*60)
        
        self.running = True
        
        # 启动视频采集
        self.video_capture.start()
        time.sleep(0.5)  # 等待摄像头稳定
        
        # 启动rPPG处理
        self.rppg_processor.start()
        
        print("="*60)
        print("✅ 所有线程已启动,系统运行中...")
        print("="*60)
    
    def stop(self):
        """停止所有线程"""
        print("\n正在停止系统...")
        self.running = False
        self.rppg_processor.stop()
        self.video_capture.stop()
        print("✅ 系统已停止")
    
    def get_status(self):
        """获取系统状态"""
        return {
            'video_fps': self.video_capture.actual_fps,
            'current_bpm': self.rppg_processor.current_bpm,
            'running': self.running
        }


def main():
    # 默认使用RDK摄像头,如果不可用则自动降级到普通摄像头
    use_rdk = True
    
    # 创建监控器
    monitor = HeartbeatMonitor(use_rdk=use_rdk)
    
    # 【关键】将视频采集器注入到web_stream,使其能直接获取原始帧
    web_stream.set_video_capture(monitor.video_capture)
    
    # 启动所有线程
    monitor.start()
    
    # 启动Web服务器(在主线程中运行)
    try:
        print("\n[INFO] 启动Web服务器...")
        print("[INFO] 访问 http://YOUR_IP:5000 查看实时监控")
        print("[INFO] 按 Ctrl+C 停止服务\n")
        
        web_stream.start_web_server()
    except KeyboardInterrupt:
        print("\n收到停止信号...")
        monitor.stop()
    except Exception as e:
        print(f"[ERROR] Web服务器错误: {e}")
        monitor.stop()


if __name__ == "__main__":
    main()