import os
import csv
import math
import random
import shutil
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import resample_poly
from math import gcd

# =========================
# 1. 路径与增强配置
# =========================

CLEAN_DIR = Path(r"your_Dataset_PATH")
OUT_DIR = Path(r"your_Dataset_PATH")

# 可以同时启用或只启用一种背景/前景噪音
# 留空列表或设置为 None 则不使用该类增强

BG_NOISE_DIR = Path(r"your_Dataset_PATH")  # 长时平稳背景噪
FG_NOISE_DIR = Path(r"your_Dataset_PATH")   # 瞬态阵发前景噪

BG_SNR_LIST = [5, 10, 15]  # 背景噪声 SNR
FG_SNR_LIST = [0, 5, 10]   # 前景噪声 SNR

# 允许覆盖的背景噪声数量 (如果只想单层噪音设为 [1])
NUM_BG_NOISES = [1]

# 两个前段噪声音频之间的最小间隔秒数
FG_INTERVAL_SEC = 2.0

TARGET_SR = 16000
RANDOM_SEED = 2026
NORMALIZE_OUTPUT = True
COPY_LABEL_TXT = True

# =========================
# 2. 基础音频工具
# =========================
def audio_to_float32(data):
    if data.dtype == np.int16:
        return data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        return data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        return (data.astype(np.float32) - 128.0) / 128.0
    else:
        data = data.astype(np.float32)
        max_abs = np.max(np.abs(data)) if data.size > 0 else 0.0
        if max_abs > 1.5:
            data = data / max_abs
        return data

def float32_to_int16(data):
    data = np.clip(data, -1.0, 1.0)
    return (data * 32767.0).astype(np.int16)

def read_wav_mono(path, target_sr=16000):
    sr, data = wavfile.read(str(path))
    if data.size == 0:
        raise ValueError(f"Empty audio file: {path}")
    data = audio_to_float32(data)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if sr != target_sr:
        g = gcd(sr, target_sr)
        data = resample_poly(data, target_sr // g, sr // g).astype(np.float32)
        sr = target_sr
    return sr, data.astype(np.float32)

def signal_power(x, eps=1e-12):
    if len(x) == 0: return eps
    return max(float(np.mean(x ** 2)), eps)

def active_speech_power(x, sr=16000, frame_ms=25, hop_ms=10, top_db=40, eps=1e-12):
    frame_len = int(sr * frame_ms / 1000.0)
    hop_len = int(sr * hop_ms / 1000.0)
    if len(x) < frame_len: return signal_power(x, eps)
    frame_powers = np.asarray([np.mean(x[s:s+frame_len]**2) for s in range(0, len(x) - frame_len + 1, hop_len)], dtype=np.float32)
    max_power = np.max(frame_powers)
    if max_power <= eps: return signal_power(x, eps)
    active = frame_powers[frame_powers >= max_power * (10 ** (-top_db / 10.0))]
    return max(float(np.mean(active)) if len(active) > 0 else signal_power(x, eps), eps)

# =========================
# 3. 增强核心引擎：生成混音画布
# =========================

def get_noise_scale(clean_pwr, noise_audio, snr_db):
    """计算单个噪音音轨需要的增益系数"""
    noise_pwr = signal_power(noise_audio)
    target_pwr = clean_pwr / (10.0 ** (snr_db / 10.0))
    return math.sqrt(target_pwr / noise_pwr) if noise_pwr > 1e-12 else 0.0

def generate_augmented_audio(clean_audio, bg_files, fg_files, rng):
    """
    根据给定的背景和前景噪声池，动态混合出一条复合噪声，并叠在 clean 上
    """
    clean_len = len(clean_audio)
    clean_pwr = active_speech_power(clean_audio, sr=TARGET_SR)
    noise_canvas = np.zeros(clean_len, dtype=np.float32)
    
    used_bg = []
    used_fg = []

    # 1. 混合背景噪声（长时稳态音）
    if bg_files and BG_SNR_LIST:
        num_bg = rng.choice(NUM_BG_NOISES)
        bg_pool = []
        for _ in range(num_bg):
            path = rng.choice(bg_files)
            try:
                _, bg_audio = read_wav_mono(path, TARGET_SR)
                if len(bg_audio) == 0: continue
            except Exception: continue
            snr = rng.choice(BG_SNR_LIST)
            
            # 从随机起始终点循环截取
            bg_len = len(bg_audio)
            start = rng.randint(0, bg_len - 1) if bg_len > 1 else 0
            indices = (np.arange(clean_len) + start) % bg_len
            bg_seg = bg_audio[indices].astype(np.float32)
            
            scale = get_noise_scale(clean_pwr, bg_seg, snr)
            bg_pool.append(bg_seg * scale)
            used_bg.append(f"{path.name}@{snr}dB")
            
        for b in bg_pool:
            noise_canvas += b

    # 2. 混合前景噪声（瞬态阵发噪音）
    if fg_files and FG_SNR_LIST:
        idx = 0
        interval_samples = int(FG_INTERVAL_SEC * TARGET_SR)
        while idx < clean_len:
            path = rng.choice(fg_files)
            try:
                _, fg_audio = read_wav_mono(path, TARGET_SR)
                if len(fg_audio) == 0:
                    idx += interval_samples
                    continue
            except Exception:
                idx += interval_samples
                continue
                
            snr = rng.choice(FG_SNR_LIST)
            fg_len = len(fg_audio)
            end_idx = min(idx + fg_len, clean_len)
            insert_len = end_idx - idx
            
            if insert_len > 0:
                fg_seg = fg_audio[:insert_len]
                scale = get_noise_scale(clean_pwr, fg_seg, snr)
                noise_canvas[idx:end_idx] += fg_seg * scale
                used_fg.append(f"{path.name}@{snr}dB_pos{idx}")
            
            idx += fg_len + interval_samples

    # 叠加
    noisy = clean_audio + noise_canvas
    peak = float(np.max(np.abs(noisy)))
    if NORMALIZE_OUTPUT and peak > 0.999:
        noisy = noisy / peak * 0.999

    desc = {
        'bg_tracks': '|'.join(used_bg) if used_bg else 'none',
        'fg_tracks': '|'.join(used_fg) if used_fg else 'none'
    }
    
    return noisy.astype(np.float32), desc

# =========================
# 4. 主处理函数
# =========================
def main():
    rng = random.Random(RANDOM_SEED)

    clean_files = sorted(CLEAN_DIR.rglob("*.wav"))
    if not clean_files:
        raise FileNotFoundError(f"未找到语音文件: {CLEAN_DIR}")

    bg_files = sorted(BG_NOISE_DIR.rglob("*.wav")) if BG_NOISE_DIR and BG_NOISE_DIR.exists() else []
    fg_files = sorted(FG_NOISE_DIR.rglob("*.wav")) if FG_NOISE_DIR and FG_NOISE_DIR.exists() else []
    
    print(f"找到干净语音: {len(clean_files)}")
    print(f"找到背景噪声: {len(bg_files)}")
    print(f"找到前景噪声: {len(fg_files)}")

    if not bg_files and not fg_files:
        print("警告: 没有任何背景或前景噪声。仅复制音频。")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = OUT_DIR / "metadata_augmented.csv"

    with open(meta_path, "w", newline="", encoding="utf-8-sig") as f_meta:
        writer = csv.writer(f_meta)
        writer.writerow(["output_wav", "clean_wav", "bg_tracks_snr", "fg_tracks_snr_pos", "sample_rate", "num_samples"])

        for idx, clean_path in enumerate(clean_files, 1):
            print(f"[{idx}/{len(clean_files)}] Processing: {clean_path.name}")
            try:
                _, clean = read_wav_mono(clean_path, TARGET_SR)
                if len(clean) == 0: continue
            except Exception as e:
                print(f"跳过 {clean_path} ({e})")
                continue

            noisy_audio, desc = generate_augmented_audio(clean, bg_files, fg_files, rng)

            rel_path = clean_path.relative_to(CLEAN_DIR)
            out_wav_path = OUT_DIR / rel_path
            out_wav_path.parent.mkdir(parents=True, exist_ok=True)

            wavfile.write(str(out_wav_path), TARGET_SR, float32_to_int16(noisy_audio))

            if COPY_LABEL_TXT:
                label_path = clean_path.with_suffix(".txt")
                if label_path.exists():
                    shutil.copy2(label_path, out_wav_path.with_suffix(".txt"))

            writer.writerow([str(out_wav_path), str(clean_path), desc['bg_tracks'], desc['fg_tracks'], TARGET_SR, len(clean)])

    print(f"完成！输出至 {OUT_DIR}")

if __name__ == "__main__":
    main()
