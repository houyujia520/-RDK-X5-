#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import argparse
import threading
import queue
import cv2
import numpy as np
from flask import Flask, Response, jsonify, redirect, request

# 导入功能模块
from OCR import zhOCR
from detect import zhdetect
from detect.utils import draw_utils as draw
from detect.utils import common_utils as common
from heartbeat import RealTimeRPPG   # 心率检测模块

# ---------- 硬件加速摄像头 ----------
try:
    from hobot_vio import libsrcampy
    HAS_HOBOT_VIO = True
    print("[INFO] hobot_vio library loaded")
except ImportError:
    HAS_HOBOT_VIO = False
    print("[WARNING] hobot_vio not found, fallback to OpenCV")

# ============================================================
# RDKCamera 类（摄像头封装）
# ============================================================
class RDKCamera:
    def __init__(self, camera_id=0, width=640, height=480, fps=30):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cam = None
        self.cap = None
        if HAS_HOBOT_VIO:
            self._init_hobot_camera()
        else:
            self._init_opencv_camera()

    def _init_hobot_camera(self):
        try:
            self.cam = libsrcampy.Camera()
            ret = self.cam.open_cam(self.camera_id, -1, self.fps, self.width, self.height)
            if ret != 0:
                raise RuntimeError(f"open_cam failed: {ret}")
            time.sleep(1)
            test_img = self.cam.get_img(2)
            if test_img is None:
                print("[WARNING] test frame read failed")
            print(f"[SUCCESS] Hobot camera {self.camera_id} {self.width}x{self.height}@{self.fps}fps")
        except Exception as e:
            print(f"[ERROR] Hobot init: {e}, fallback OpenCV")
            self._init_opencv_camera()

    def _init_opencv_camera(self):
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_id}")
        print(f"[SUCCESS] OpenCV camera {self.width}x{self.height}@{self.fps}fps")

    def read_frame(self):
        if HAS_HOBOT_VIO and self.cam:
            try:
                img_data = self.cam.get_img(2)
                if img_data is None:
                    return False, None
                img_array = np.frombuffer(img_data, dtype=np.uint8)
                frame = img_array.reshape((self.height + self.height // 2, self.width))
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_NV21)
                return True, frame_bgr
            except Exception as e:
                print(f"[ERROR] Frame read: {e}")
                return False, None
        else:
            return self.cap.read()

    def release(self):
        if self.cam:
            self.cam.close_cam()
            self.cam = None
        if self.cap:
            self.cap.release()
            self.cap = None
        print("[INFO] Camera released")

# ============================================================
# Flask App
# ============================================================
app = Flask(__name__)

@app.after_request
def after_request(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

# ============================================================
# 全局共享状态
# ============================================================
lock = threading.Lock()
stop_event = threading.Event()

latest_frame = None
latest_pose_results = {
    "people_count": 0,
    "detections": []
}
latest_ocr_result = {
    "texts": [],
    "matched_name": None,
    "matched_advice": None,
    "timestamp": 0
}
latest_heart_rate = {
    "bpm": 0.0,
    "sqi": 0.0,
    "avg3s_hr": 0.0
}
current_mode = 'pose'   # 'pose' | 'ocr' | 'heartbeat'

# OCR 专用队列
ocr_queue = queue.Queue(maxsize=1)

# 心率检测器实例（懒加载）
heartbeat_detector = None

# ============================================================
# OCR 后台工作线程
# ============================================================
def ocr_worker():
    global latest_ocr_result
    while not stop_event.is_set():
        try:
            frame = ocr_queue.get(timeout=0.5)
            if frame is None:
                continue
            try:
                raw = zhOCR.reader.readtext(
                    frame,
                    paragraph=False,
                    min_size=10,
                    text_threshold=0.6,
                    low_text=0.3
                )
                filtered = zhOCR.filter_drug_text(raw)
                matched_name, matched_advice = zhOCR.match_drug_advice(filtered)

                with lock:
                    latest_ocr_result = {
                        "texts": [item['text'] for item in filtered],
                        "matched_name": matched_name,
                        "matched_advice": matched_advice,
                        "timestamp": time.time()
                    }
            except Exception as e:
                print(f"[OCR Worker Error] {e}")
            ocr_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[OCR Worker Exception] {e}")

# ============================================================
# 主推理循环（支持三种模式）
# ============================================================
def inference_worker(camera, pose_model, opt):
    global latest_frame, latest_pose_results, latest_ocr_result, latest_heart_rate
    global current_mode, heartbeat_detector

    frame_count = 0
    max_frames = opt.max_frames
    ocr_skip_counter = 0
    OCR_PROCESS_INTERVAL = 10
    last_drawn_ocr = {"name": None, "advice": None}
    
    # 心率模式专用计数器
    heartbeat_process_counter = 0
    HEARTBEAT_PROCESS_INTERVAL = 2  # 每2帧处理一次心率（提高采样率）

    while not stop_event.is_set():
        if max_frames > 0 and frame_count >= max_frames:
            print(f"[INFO] Reached max frames ({max_frames}), stopping worker.")
            break

        ret, frame = camera.read_frame()
        if not ret or frame is None:
            time.sleep(0.02)
            continue

        img_h, img_w = frame.shape[:2]
        mode = current_mode

        if mode == 'pose':
            # ---------- 姿态检测 ----------
            ids, scores, boxes, kpts_xy, kpt_score = zhdetect.infer_pose(pose_model, frame, img_h, img_w)
            try:
                draw.draw_boxes(frame, boxes, ids, scores, ["person"], common.rdk_colors)
                draw.draw_keypoints(frame, kpts_xy, kpt_score, kpt_conf_thresh=opt.kpt_conf_thres)
            except Exception as e:
                print(f"[WARN] Drawing failed: {e}")

            detections = []
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                h = y2 - y1
                w = x2 - x1
                ratio = w / h if h > 0 else 0
                is_fall = ratio > opt.fall_threshold
                cv2.putText(frame, "FALL" if is_fall else "STANDING",
                            (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0,0,255) if is_fall else (0,255,0), 2)
                detections.append({
                    "bbox": [float(v) for v in boxes[i]],
                    "keypoints": [[float(p[0]), float(p[1])] for p in kpts_xy[i]],
                    "keypoint_scores": [float(s) for s in kpt_score[i]],
                    "score": float(scores[i]),
                    "fall": bool(is_fall)
                })
            with lock:
                latest_frame = frame.copy()
                latest_pose_results = {"people_count": len(boxes), "detections": detections}

        elif mode == 'ocr':
            # ---------- OCR 模式 ----------
            ocr_skip_counter += 1
            if ocr_skip_counter % OCR_PROCESS_INTERVAL == 0:
                target_width = 240
                scale = target_width / img_w
                dim = (target_width, int(img_h * scale))
                resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
                try:
                    ocr_queue.put_nowait(resized)
                except queue.Full:
                    try:
                        ocr_queue.get_nowait()
                        ocr_queue.put_nowait(resized)
                    except queue.Empty:
                        pass

            with lock:
                current_name = latest_ocr_result["matched_name"]
                current_advice = latest_ocr_result["matched_advice"]
            if current_name != last_drawn_ocr["name"] or current_advice != last_drawn_ocr["advice"]:
                last_drawn_ocr["name"] = current_name
                last_drawn_ocr["advice"] = current_advice

            if last_drawn_ocr["name"]:
                cv2.putText(frame, f"Drug: {last_drawn_ocr['name']}", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                cv2.putText(frame, f"Advice: {last_drawn_ocr['advice']}", (10,60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            else:
                cv2.putText(frame, "No drug detected", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            with lock:
                latest_frame = frame.copy()

        elif mode == 'heartbeat':
            # ---------- 心率检测 ----------
            if heartbeat_detector is None:
                # 首次使用，初始化心率检测器（使用 opt 中的参数）
                print("[INFO] 初始化心率检测器...")
                heartbeat_detector = RealTimeRPPG(opt)
            
            # 每帧调用process_frame，内部会自动跳过奇数帧（与参考代码一致）
            bpm, sqi, avg3s = heartbeat_detector.process_frame(frame)
            
            # 在视频帧上绘制心率
            with lock:
                latest_heart_rate["bpm"] = bpm
                latest_heart_rate["sqi"] = sqi
                latest_heart_rate["avg3s_hr"] = avg3s
            
            if bpm > 0:
                cv2.putText(frame, f"HR: {bpm:.1f} BPM", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.putText(frame, f"SQI: {sqi:.2f}", (10,60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            else:
                cv2.putText(frame, "Detecting HR...", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,165,255), 2)
                cv2.putText(frame, "Please keep face visible", (10,60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)
            
            with lock:
                latest_frame = frame.copy()

        frame_count += 1
        time.sleep(0.02)

    print("[INFO] Inference worker stopped.")

# ============================================================
# Flask 端点
# ============================================================
def generate_video_stream():
    while not stop_event.is_set():
        frame = None
        with lock:
            if latest_frame is not None:
                frame = latest_frame.copy()
        if frame is None:
            time.sleep(0.02)
            continue

        try:
            small = cv2.resize(frame, (480, 270))
            small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            ret, jpeg = cv2.imencode('.jpg', small_rgb, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   jpeg.tobytes() + b'\r\n')
        except Exception as e:
            print(f"[WARN] generate_video_stream: {e}")
            time.sleep(0.02)

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/mode', methods=['GET', 'POST'])
def mode_switch():
    global current_mode
    if request.method == 'POST':
        data = request.get_json()
        new_mode = data.get('mode', '').lower()
        if new_mode in ['pose', 'ocr', 'heartbeat']:
            current_mode = new_mode
            print(f"[INFO] Mode switched to {new_mode}")
            return jsonify({"status": "ok", "mode": current_mode})
        else:
            return jsonify({"error": "Invalid mode"}), 400
    else:
        return jsonify({"mode": current_mode})

@app.route('/api/status')
def api_status():
    mode = current_mode
    with lock:
        if mode == 'pose':
            data = latest_pose_results.copy()
        elif mode == 'ocr':
            data = {
                "people_count": 0,
                "detections": [],
                "ocr": {
                    "texts": latest_ocr_result.get("texts", []),
                    "matched_name": latest_ocr_result.get("matched_name"),
                    "matched_advice": latest_ocr_result.get("matched_advice"),
                    "timestamp": latest_ocr_result.get("timestamp", 0)
                }
            }
        else:  # heartbeat
            data = {
                "people_count": 0,
                "detections": [],
                "heart_rate": {
                    "bpm": latest_heart_rate.get("bpm", 0.0),
                    "sqi": latest_heart_rate.get("sqi", 0.0),
                    "avg3s_hr": latest_heart_rate.get("avg3s_hr", 0.0)
                }
            }
    return jsonify({"mode": mode, **data})

@app.route('/health')
def health():
    return {"status": "running"}

@app.route('/')
def index():
    return redirect('/static/zhenghe.html')

# ============================================================
# 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    # 姿态参数
    parser.add_argument('--model-path', type=str,
                        default='detect/yolo11n_pose_bayese_640x640_nv12.bin',
                        help='BPU model path')
    parser.add_argument('--priority', type=int, default=0)
    parser.add_argument('--bpu-cores', nargs='+', type=int, default=[0])
    parser.add_argument('--source', type=str, default='0',
                        help='camera index')
    parser.add_argument('--cam-width', type=int, default=1920)
    parser.add_argument('--cam-height', type=int, default=1080)
    parser.add_argument('--cam-fps', type=int, default=30)
    parser.add_argument('--score-thres', type=float, default=0.25)
    parser.add_argument('--kpt-conf-thres', type=float, default=0.5)
    parser.add_argument('--fall-threshold', type=float, default=0.85)
    parser.add_argument('--max-frames', type=int, default=-1)
    # 心率参数
    parser.add_argument('--heartbeat_model', type=str,
                        default='heartbeat/models/haarcascade_frontalface_alt.xml',
                        help='心率模型路径')
    parser.add_argument('--hr_min', type=float, default=40.0)
    parser.add_argument('--hr_max', type=float, default=220.0)
    parser.add_argument('--window_size', type=int, default=300,
                        help='分析窗口大小（帧数）')
    parser.add_argument('--use_gpu', action='store_true', default=False,
                        help='使用GPU加速')
    # Flask 参数
    parser.add_argument('--port', type=int, default=5000,
                        help='Flask port')
    opt = parser.parse_args()

    if not os.path.exists(opt.model_path):
        print(f"[ERROR] 姿态模型 {opt.model_path} 不存在")
        sys.exit(1)
    
    # 检查心率模型（可选，因为 RealTimeRPPG 内部会处理）
    heartbeat_model_path = opt.heartbeat_model.replace('\\', '/')
    if not os.path.exists(heartbeat_model_path):
        print(f"[WARNING] 心率模型 {heartbeat_model_path} 不存在，心率功能可能不可用")

    # 初始化 OCR
    zhOCR.init_ocr()

    # 初始化姿态模型
    pose_model = zhdetect.init_pose_model(opt)

    # 初始化摄像头
    camera = RDKCamera(camera_id=int(opt.source) if opt.source.isdigit() else 0,
                       width=opt.cam_width, height=opt.cam_height, fps=opt.cam_fps)

    # 启动 OCR 线程
    ocr_thread = threading.Thread(target=ocr_worker, daemon=True)
    ocr_thread.start()

    # 启动主推理线程
    worker_thread = threading.Thread(target=inference_worker,
                                     args=(camera, pose_model, opt),
                                     daemon=True)
    worker_thread.start()

    print(f"[INFO] 服务启动于 http://0.0.0.0:{opt.port}")
    print(f"[INFO] 支持模式: pose, ocr, heartbeat")
    try:
        app.run(host='0.0.0.0', port=opt.port, threaded=True)
    except KeyboardInterrupt:
        print("[INFO] 正在关闭...")
    finally:
        stop_event.set()
        worker_thread.join(timeout=2.0)
        ocr_thread.join(timeout=2.0)
        camera.release()
        print("[INFO] 完成")

if __name__ == '__main__':
    main()