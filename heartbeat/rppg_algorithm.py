import numpy as np
from scipy.signal import detrend, butter, filtfilt, find_peaks
import time

class RPPGProcessor:
    def __init__(self, fs=30, min_signal_size=100, max_signal_size=300):
        self.fs = fs
        self.min_signal_size = min_signal_size
        self.max_signal_size = max_signal_size
        self.raw_signals = [] 
        self.buffer_size = max_signal_size
        
        # 用于平滑去趋势的移动平均窗口
        self.detrend_window = max(10, int(fs * 1.0)) # 至少1秒的窗口

    def extract_raw_signal(self, roi):
        """
        从ROI中提取平均RGB值，并剔除过亮/过暗像素以抵抗光照干扰
        """
        if roi.size == 0:
            return np.array([0, 0, 0])
            
        # 转换为浮点数处理
        roi_float = roi.astype(np.float64)
        
        # 计算每个通道的均值和标准差
        mean_rgb = np.mean(roi_float, axis=(0, 1))
        
        # 【优化】剔除异常像素：忽略超过均值+2标准差的像素（高光反射）
        # 这有助于减少灯光直射造成的干扰
        std_rgb = np.std(roi_float, axis=(0, 1))
        mask = np.all(roi_float < (mean_rgb + 2 * std_rgb), axis=2)
        
        if np.sum(mask) > 10: # 确保有足够的像素
            clean_roi = roi_float[mask]
            mean_rgb_clean = np.mean(clean_roi, axis=0)
            return mean_rgb_clean
        else:
            return mean_rgb

    def _moving_average_detrend(self, signal, window_size=None):
        """
        使用移动平均进行去趋势，比线性去趋势更能适应光照非线性变化
        """
        if window_size is None:
            window_size = self.detrend_window
            
        kernel = np.ones(window_size) / window_size
        trend = np.convolve(signal, kernel, mode='same')
        
        # 处理边界效应
        half_w = window_size // 2
        trend[:half_w] = trend[half_w]
        trend[-half_w:] = trend[-half_w-1]
        
        return signal - trend

    def process_frame(self, raw_rgb):
        """
        添加新帧数据并尝试计算心率
        返回: (heart_rate, valid)
        """
        self.raw_signals.append(raw_rgb)
        
        if len(self.raw_signals) > self.buffer_size:
            self.raw_signals.pop(0)
            
        if len(self.raw_signals) < self.min_signal_size:
            return 0, False
        
        signals = np.array(self.raw_signals).T # (3, N)
        
        # --- 步骤 1: 去趋势 (使用移动平均) ---
        # 对每个通道分别去趋势
        signals_detrended = np.apply_along_axis(self._moving_average_detrend, 1, signals)
        
        # --- 步骤 2: 标准化 ---
        std_dev = np.std(signals_detrended, axis=1, keepdims=True)
        # 【优化】如果标准差太小，说明信号太弱（可能光照不足或无面部），标记为无效
        if np.any(std_dev < 1e-4):
            return 0, False
            
        normalized = signals_detrended / std_dev
        
        R, G, B = normalized[0], normalized[1], normalized[2]
        
        # --- 步骤 3: POS 算法 ---
        # 重新归一化每个通道相对于其均值
        # 注意：这里使用原始信号的均值来计算归一化因子，而不是去趋势后的
        original_signals = np.array(self.raw_signals).T
        R_mean = np.mean(original_signals[0])
        G_mean = np.mean(original_signals[1])
        B_mean = np.mean(original_signals[2])
        
        if R_mean < 1e-6 or G_mean < 1e-6 or B_mean < 1e-6:
            return 0, False
            
        rn = R / (R_mean + 1e-6)
        gn = G / (G_mean + 1e-6)
        bn = B / (B_mean + 1e-6)
        
        # POS 投影
        Xs = 3 * rn - 2 * gn
        Ys = 1.5 * rn + gn - 1.5 * bn
        
        # --- 步骤 4: 带通滤波 ---
        lowcut = 1.0  # 42 BPM
        highcut = 3.5 # 210 BPM
        
        nyq = 0.5 * self.fs
        if nyq <= highcut:
            return 0, False
            
        low = lowcut / nyq
        high = highcut / nyq
        
        if low >= high:
            return 0, False

        b, a = butter(3, [low, high], btype='band')
        
        try:
            X_filt = filtfilt(b, a, Xs)
            Y_filt = filtfilt(b, a, Ys)
            
            # 合成信号 S
            alpha = np.std(X_filt) / (np.std(Y_filt) + 1e-6)
            S = X_filt - alpha * Y_filt
            
            # --- 步骤 5: 计算心率 ---
            
            # 1. 频域 FFT
            fft_vals = np.fft.rfft(S)
            fft_freqs = np.fft.rfftfreq(len(S), 1.0/self.fs)
            
            mask = (fft_freqs >= lowcut) & (fft_freqs <= highcut)
            if not np.any(mask):
                return 0, False
                
            freqs = fft_freqs[mask]
            mags = np.abs(fft_vals[mask])
            
            # 【优化】寻找前两个最大峰值，检查是否有谐波干扰
            peak_indices = np.argsort(mags)[-2:][::-1] # 降序排列前两个
            peak_freq_fft = freqs[peak_indices[0]]
            bpm_fft = peak_freq_fft * 60.0
            
            # 如果最大峰值的能量不够显著，认为信号不可信
            max_mag = mags[peak_indices[0]]
            noise_floor = np.median(mags)
            if max_mag < 2 * noise_floor: # 信噪比阈值
                return 0, False

            # 2. 时域峰值检测
            # 【优化】根据当前估算的心率动态调整最小距离
            min_dist_seconds = 60.0 / (bpm_fft + 20) # 允许比当前心率快20bpm
            distance_samples = int(self.fs * min_dist_seconds)
            if distance_samples < 5: distance_samples = 5
            
            peaks, properties = find_peaks(S, distance=distance_samples, height=0)
            
            bpm_time = 0
            valid_time = False
            
            if len(peaks) > 2:
                peak_times = peaks / self.fs
                intervals = np.diff(peak_times[-4:]) # 取最近4个间隔
                
                if len(intervals) > 0:
                    avg_interval = np.mean(intervals)
                    std_interval = np.std(intervals)
                    
                    # 变异系数检查：如果间隔变化太大，说明检测不准
                    if std_interval / avg_interval < 0.15 and 0.25 < avg_interval < 1.5:
                        bpm_time = 60.0 / avg_interval
                        valid_time = True
            
            # 3. 融合策略
            final_bpm = bpm_fft
            if valid_time:
                # 如果时域和频域接近，优先使用时域（响应快）
                if abs(bpm_time - bpm_fft) < 8:
                    final_bpm = bpm_time
                # 如果频域极低（常见错误），使用时域
                elif bpm_fft < 45:
                    final_bpm = bpm_time

            return final_bpm, True
            
        except Exception as e:
            print(f"Signal processing error: {e}")
            return 0, False
