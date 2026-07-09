"""信号处理模块 - 心率计算和后处理"""

import numpy as np
import scipy.signal
from scipy.signal import butter, filtfilt


class SignalProcessor:
    """rPPG信号处理器"""
    
    def __init__(self, fs=30):
        """
        初始化信号处理器
        
        Args:
            fs: 采样频率 (帧率)
        """
        self.fs = fs
        
    def detrend(self, signal, lambda_value=100):
        """
        去趋势处理
        
        Args:
            signal: 输入信号
            lambda_value: 平滑参数
            
        Returns:
            去趋势后的信号
        """
        from scipy.sparse import spdiags
        
        signal_length = len(signal)
        if signal_length < 3:
            return signal
        
        # 构造二阶差分矩阵
        ones = np.ones(signal_length)
        minus_twos = -2 * np.ones(signal_length)
        diags_data = np.array([ones, minus_twos, ones])
        diags_index = np.array([0, 1, 2])
        
        D = spdiags(diags_data, diags_index, 
                   signal_length - 2, signal_length).toarray()
        
        # 计算去趋势信号
        H = np.identity(signal_length)
        detrended = np.dot(
            (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))),
            signal
        )
        
        return detrended
    
    def bandpass_filter(self, signal, lowcut=0.6, highcut=3.3, order=1):
        """
        带通滤波
        
        Args:
            signal: 输入信号
            lowcut: 低频截止 (Hz)
            highcut: 高频截止 (Hz)
            order: 滤波器阶数
            
        Returns:
            滤波后的信号
        """
        nyquist = 0.5 * self.fs
        low = lowcut / nyquist
        high = highcut / nyquist
        
        b, a = butter(order, [low, high], btype='band')
        filtered = filtfilt(b, a, signal)
        
        return filtered
    
    def calculate_hr_fft(self, ppg_signal, low_pass=0.6, high_pass=3.3):
        """
        使用FFT方法计算心率
        
        Args:
            ppg_signal: PPG信号
            low_pass: 最低频率 (Hz)
            high_pass: 最高频率 (Hz)
            
        Returns:
            心率值 (BPM)
        """
        if len(ppg_signal) < self.fs:  # 至少需要1秒数据
            return 0.0
        
        # 去趋势
        detrended = self.detrend(ppg_signal)
        
        # 带通滤波
        filtered = self.bandpass_filter(detrended, low_pass, high_pass)
        
        # 计算下一个2的幂次
        N = self._next_power_of_2(len(filtered))
        
        # 计算周期图
        frequencies, psd = scipy.signal.periodogram(
            filtered, fs=self.fs, nfft=N, detrend=False
        )
        
        # 找到感兴趣频率范围内的峰值
        freq_mask = (frequencies >= low_pass) & (frequencies <= high_pass)
        freqs_in_range = frequencies[freq_mask]
        psd_in_range = psd[freq_mask]
        
        if len(psd_in_range) == 0:
            return 0.0
        
        # 找到最大功率对应的频率
        max_idx = np.argmax(psd_in_range)
        dominant_freq = freqs_in_range[max_idx]
        
        # 转换为BPM
        hr_bpm = dominant_freq * 60.0
        
        return hr_bpm
    
    def calculate_hr_peak(self, ppg_signal):
        """
        使用峰值检测方法计算心率
        
        Args:
            ppg_signal: PPG信号
            
        Returns:
            心率值 (BPM)
        """
        if len(ppg_signal) < self.fs * 2:  # 至少需要2秒数据
            return 0.0
        
        # 去趋势
        detrended = self.detrend(ppg_signal)
        
        # 带通滤波
        filtered = self.bandpass_filter(detrended)
        
        # 寻找峰值
        peaks, _ = scipy.signal.find_peaks(filtered, distance=self.fs*0.3)
        
        if len(peaks) < 2:
            return 0.0
        
        # 计算平均峰间间隔
        peak_intervals = np.diff(peaks)
        avg_interval = np.mean(peak_intervals)
        
        # 转换为BPM
        hr_bpm = 60.0 / (avg_interval / self.fs)
        
        return hr_bpm
    
    def calculate_snr(self, ppg_signal, hr_bpm):
        """
        计算信噪比
        
        Args:
            ppg_signal: PPG信号
            hr_bpm: 心率值 (BPM)
            
        Returns:
            SNR值 (dB)
        """
        if len(ppg_signal) < self.fs:
            return 0.0
        
        # 去趋势和滤波
        detrended = self.detrend(ppg_signal)
        filtered = self.bandpass_filter(detrended)
        
        # 计算FFT
        N = self._next_power_of_2(len(filtered))
        frequencies, psd = scipy.signal.periodogram(
            filtered, fs=self.fs, nfft=N, detrend=False
        )
        
        # 基频和谐波
        fundamental_freq = hr_bpm / 60.0
        second_harmonic = 2 * fundamental_freq
        deviation = 6 / 60.0  # 6 BPM的容差
        
        # 找到各个频段的索引
        idx_fundamental = np.where(
            (frequencies >= fundamental_freq - deviation) & 
            (frequencies <= fundamental_freq + deviation)
        )[0]
        
        idx_harmonic = np.where(
            (frequencies >= second_harmonic - deviation) & 
            (frequencies <= second_harmonic + deviation)
        )[0]
        
        idx_noise = np.where(
            (frequencies >= 0.6) & (frequencies <= 3.3) &
            ~((frequencies >= fundamental_freq - deviation) & 
              (frequencies <= fundamental_freq + deviation)) &
            ~((frequencies >= second_harmonic - deviation) & 
              (frequencies <= second_harmonic + deviation))
        )[0]
        
        # 计算功率
        signal_power = np.sum(psd[idx_fundamental]) + np.sum(psd[idx_harmonic])
        noise_power = np.sum(psd[idx_noise])
        
        if noise_power == 0:
            return 0.0
        
        # 计算SNR (dB)
        snr = 10 * np.log10(signal_power / noise_power)
        
        return snr
    
    @staticmethod
    def _next_power_of_2(x):
        """计算大于等于x的最小2的幂"""
        return 1 if x == 0 else 2 ** (x - 1).bit_length()
    
    def smooth_signal(self, signal, window_size=5):
        """
        平滑信号
        
        Args:
            signal: 输入信号
            window_size: 滑动窗口大小
            
        Returns:
            平滑后的信号
        """
        if len(signal) < window_size:
            return signal
        
        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(signal, kernel, mode='same')
        
        return smoothed
