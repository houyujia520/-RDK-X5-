import os

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HAAR_CASCADE_PATH = os.path.join(BASE_DIR, "models", "haarcascade_frontalface_alt.xml")

# 算法参数
DEFAULT_DOWNSAMPLE = 1
SAMPLING_FREQUENCY = 30  # Hz
RESCAN_FREQUENCY = 10    # 每多少帧重新检测一次人脸
MIN_SIGNAL_SIZE = 100    # 信号处理最小窗口
MAX_SIGNAL_SIZE = 300    # 信号处理最大窗口

# 视频源 (0 为摄像头，或填写视频文件路径)
VIDEO_SOURCE = 0 