import os
from pathlib import Path
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import resample_poly
from math import gcd

DATA_DIR = Path(r"your_Dataset_PATH")
TARGET_SR = 16000

def audio_to_float32(data):
    if data.dtype == np.int16:
        return data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        return data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        return (data.astype(np.float32) - 128.0) / 128.0
    else:
        return data.astype(np.float32)

def float32_to_int16(data):
    data = np.clip(data, -1.0, 1.0)
    return (data * 32767.0).astype(np.int16)

def process_audio(file_path):
    try:
        sr, data = wavfile.read(str(file_path))
        if sr == TARGET_SR and data.ndim == 1:
            return True
            
        data = audio_to_float32(data)
        
        # 1. 转换为单声道 (取双通道平均)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
            
        # 2. 降采样 (48000 -> 16000)
        if sr != TARGET_SR:
            g = gcd(sr, TARGET_SR)
            data = resample_poly(data, TARGET_SR // g, sr // g).astype(np.float32)
            
        # 3. 覆盖原始音频文件
        wavfile.write(str(file_path), TARGET_SR, float32_to_int16(data))
        print(f"[{file_path.name}] 成功转换为单声道 16000Hz 并覆盖原文件")
        return True
    except Exception as e:
        print(f"[{file_path.name}] 处理失败: {e}")
        return False

def main():
    if not DATA_DIR.exists():
        print(f"目录不存在: {DATA_DIR}")
        return
        
    wav_files = list(DATA_DIR.rglob("*.wav"))
    print(f"找到 {len(wav_files)} 个音频，准备降采样(16kHz) + 单声道提取并覆盖...")
    
    success_count = 0
    for idx, path in enumerate(wav_files, 1):
        if process_audio(path):
            success_count += 1
            
    print(f"全部完成！成功处理: {success_count}/{len(wav_files)}")

if __name__ == '__main__':
    main()
