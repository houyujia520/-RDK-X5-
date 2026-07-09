"""
RDK摄像头可用性测试脚本
用于快速检测系统是否支持RDK硬件加速摄像头
"""

def test_rdk_camera():
    """测试RDK摄像头是否可用"""
    print("="*60)
    print("RDK摄像头可用性测试")
    print("="*60)
    
    # 1. 检查hobot_vio库
    try:
        from hobot_vio import libsrcampy
        print("✅ hobot_vio库已安装")
    except ImportError as e:
        print(f"❌ hobot_vio库未找到: {e}")
        print("\n建议:")
        print("- 如果在普通PC上运行,这是正常的,系统将使用普通摄像头")
        print("- 如果在RDK板上运行,请确保已正确安装hobot-vio库")
        return False
    
    # 2. 尝试打开摄像头
    try:
        cam = libsrcampy.Camera()
        ret = cam.open_cam(0, -1, 30, 640, 480)
        
        if ret != 0:
            print(f"❌ 无法打开RDK摄像头 (错误码: {ret})")
            return False
        
        print("✅ RDK摄像头打开成功")
        
        # 3. 尝试读取一帧
        img_data = cam.get_img(2)
        if img_data is None:
            print("❌ 无法读取图像数据")
            cam.close_cam()
            return False
        
        print(f"✅ 成功读取一帧数据 (大小: {len(img_data)} bytes)")
        
        # 4. 验证数据格式
        import numpy as np
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        total_pixels = len(img_array)
        
        # 推断分辨率
        if total_pixels == 1920 * 1080 * 3 // 2:
            resolution = "1920x1080"
        elif total_pixels == 1280 * 720 * 3 // 2:
            resolution = "1280x720"
        elif total_pixels == 640 * 480 * 3 // 2:
            resolution = "640x480"
        else:
            resolution = f"未知 ({total_pixels} pixels)"
        
        print(f"✅ 摄像头分辨率: {resolution}")
        
        # 释放资源
        cam.close_cam()
        print("✅ 摄像头已关闭")
        
        print("\n" + "="*60)
        print("测试结果: RDK摄像头完全可用!")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        return False


def test_opencv_camera():
    """测试普通OpenCV摄像头"""
    print("\n" + "="*60)
    print("普通OpenCV摄像头测试")
    print("="*60)
    
    import cv2
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ 无法打开普通摄像头")
        return False
    
    print("✅ 普通摄像头打开成功")
    
    # 获取摄像头信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"📹 分辨率: {width}x{height}")
    print(f"📹 标称FPS: {fps}")
    
    # 尝试读取一帧
    ret, frame = cap.read()
    if not ret or frame is None:
        print("❌ 无法读取图像")
        cap.release()
        return False
    
    print(f"✅ 成功读取一帧 (形状: {frame.shape})")
    
    cap.release()
    print("✅ 摄像头已关闭")
    
    print("\n" + "="*60)
    print("测试结果: 普通摄像头可用")
    print("="*60)
    return True


if __name__ == "__main__":
    # 首先测试RDK摄像头
    rdk_available = test_rdk_camera()
    
    # 然后测试普通摄像头
    opencv_available = test_opencv_camera()
    
    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    
    if rdk_available:
        print("✅ 推荐使用RDK摄像头 (硬件加速)")
    elif opencv_available:
        print("⚠️  RDK不可用,将使用普通摄像头")
    else:
        print("❌ 没有可用的摄像头!")
    
    print("="*60)