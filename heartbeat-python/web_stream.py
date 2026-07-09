# file: d:\HP\Documents\heartbeat-python\web_stream.py
from flask import Flask, Response, jsonify
from flask_cors import CORS
import cv2
import threading
import numpy as np
import time  # 引入时间模块

app = Flask(__name__)
CORS(app)

# 全局变量
latest_frame = None
lock = threading.Lock()

# 【修改】使用列表存储最近的心率数据: [(timestamp, bpm), ...]
bpm_history = [] 
# 用于缓存最近计算好的平均BPM，避免频繁计算导致API响应慢，也可实时计算
last_avg_bpm = 0 

# 视频采集器引用(由main.py注入)
video_capture_ref = None


def generate_frames():
    """
    生成器函数：直接从视频采集线程获取最新帧并编码为 JPEG 发送给前端
    【优化】不再依赖处理后的帧,直接使用原始视频帧,确保最高帧率
    """
    global latest_frame, video_capture_ref
    
    while True:
        # 【优化】优先从视频采集线程获取最新帧
        frame_copy = None
        
        if video_capture_ref is not None:
            frame_copy = video_capture_ref.get_latest_frame()
        
        # 降级方案:使用旧的全局变量
        if frame_copy is None:
            with lock:
                if latest_frame is not None:
                    frame_copy = latest_frame.copy()
        
        # 如果仍然没有帧,短暂等待
        if frame_copy is None:
            threading.Event().wait(0.01)
            continue

        # 编码为 JPEG - 降低质量以提升速度
            small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            ret, jpeg = cv2.imencode('.jpg', small_rgb, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # 【优化】控制帧率,避免推送过快导致浏览器卡顿
        # 目标30fps,每帧间隔约33ms
        time.sleep(0.03)


@app.route('/status')  
def get_status():
    """
    API 路由：返回最近3秒的平均心率
    支持 /status 和 /api/status 两种路径
    """
    global bpm_history, last_avg_bpm
    
    current_time = time.time()
    valid_bpms = []
    
    with lock:
        # 1. 清理过期数据 (保留最近3秒)
        # 保留 timestamp > current_time - 3.0 的数据
        bpm_history = [(ts, bpm) for ts, bpm in bpm_history if current_time - ts < 3.0]
        
        # 2. 提取有效的 BPM 值 (排除 0 或无效值)
        valid_bpms = [bpm for _, bpm in bpm_history if bpm > 0]
        
        # 3. 计算平均值
        if valid_bpms:
            avg_bpm = sum(valid_bpms) / len(valid_bpms)
            last_avg_bpm = avg_bpm # 更新缓存
        else:
            # 如果最近3秒没有有效数据，返回上一次的平均值，或者0
            # 这里选择返回 last_avg_bpm 以保持图表平滑，或者返回 0 表示丢失
            avg_bpm = last_avg_bpm if last_avg_bpm > 0 else 0

    return jsonify({
        'bpm': round(avg_bpm, 1), # 保留一位小数
        'count': len(valid_bpms), # 调试用：显示参与计算的数据点数量
        'status': 'online' if avg_bpm > 0 else 'searching'
    })


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/drug')  
def drug_recognition():
    """
    药品识别API (预留接口)
    当前返回空数据,未来可扩展药品识别功能
    支持 /drug 和 /api/drug 两种路径
    """
    return jsonify({
        'drug_name': None,
        'advice': None,
        'confidence': 0.0
    })



def set_video_capture(video_capture):
    """
    设置视频采集器引用(由main.py调用)
    这样web_stream可以直接从采集线程获取帧,无需等待处理
    """
    global video_capture_ref
    video_capture_ref = video_capture


def start_web_server():
    app.run(host='0.0.0.0', port=5000, threaded=True)


if __name__ == '__main__':
    pass