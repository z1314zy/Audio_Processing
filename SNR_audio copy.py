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

CLEAN_DIR = Path(r"D:\VAD\1.vad_learn\ten-vad\testset")
OUT_DIR = Path(r"D:\VAD\1.vad_learn\ten-vad\SNR_NOISE_PR_PYTHON\testset_STREET")

# 可以同时启用或只启用一种背景/前景噪音
# 留空列表或设置为 None 则不使用该类增强

BG_NOISE_DIR = Path(r"D:\VAD\1.vad_learn\data_processing\QUT_Dataset_15s\STREET")  # 背景噪声目录

BG_SNR_LIST = [5, 8, 10, 15]  # 规定输出的 SNR 列表，每个值都会生成一个对应子文件夹

# 叠加的背景噪声数量
NUM_BG_NOISES = 1

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

def generate_augmented_audio(clean_audio, bg_files, snr, rng):
    """
    给纯净语音叠加指定信噪比 (SNR) 的单一背景噪声
    """
    clean_len = len(clean_audio)
    clean_pwr = active_speech_power(clean_audio, sr=TARGET_SR)
    noise_canvas = np.zeros(clean_len, dtype=np.float32)
    
    used_bg = []

    # 混合背景噪声
    if bg_files:
        for _ in range(NUM_BG_NOISES):
            path = rng.choice(bg_files)
            try:
                _, bg_audio = read_wav_mono(path, TARGET_SR)
                if len(bg_audio) == 0: continue
            except Exception: continue
            
            # 从随机起起始点循环截取
            bg_len = len(bg_audio)
            start = rng.randint(0, bg_len - 1) if bg_len > 1 else 0
            indices = (np.arange(clean_len) + start) % bg_len
            bg_seg = bg_audio[indices].astype(np.float32)
            
            scale = get_noise_scale(clean_pwr, bg_seg, snr)
            bg_pool_mix = bg_seg * scale
            noise_canvas += bg_pool_mix
            used_bg.append(f"{path.name}@{snr}dB")

    # 叠加
    noisy = clean_audio + noise_canvas
    peak = float(np.max(np.abs(noisy)))
    if NORMALIZE_OUTPUT and peak > 0.999:
        noisy = noisy / peak * 0.999

    desc = '|'.join(used_bg) if used_bg else 'none'
    
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
    
    print(f"找到干净语音: {len(clean_files)}")
    print(f"找到背景噪声: {len(bg_files)}")

    if not bg_files:
        print("警告: 没有任何背景噪声文件，仅复制音频。")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = OUT_DIR / "metadata_augmented.csv"

    with open(meta_path, "w", newline="", encoding="utf-8-sig") as f_meta:
        writer = csv.writer(f_meta)
        writer.writerow(["output_wav", "clean_wav", "bg_tracks_snr", "snr_class", "sample_rate", "num_samples"])

        for idx, clean_path in enumerate(clean_files, 1):
            print(f"[{idx}/{len(clean_files)}] Processing: {clean_path.name}")
            try:
                _, clean = read_wav_mono(clean_path, TARGET_SR)
                if len(clean) == 0: continue
            except Exception as e:
                print(f"跳过 {clean_path} ({e})")
                continue

            rel_path = clean_path.relative_to(CLEAN_DIR)

            # 在规定的 SNR 列表里循环，分别输出到各自的文件下
            for current_snr in BG_SNR_LIST:
                noisy_audio, desc = generate_augmented_audio(clean, bg_files, current_snr, rng)

                # 创建对应 snr dB 的子文件夹
                snr_dir = OUT_DIR / f"{current_snr}dB"
                out_wav_path = snr_dir / rel_path
                out_wav_path.parent.mkdir(parents=True, exist_ok=True)

                wavfile.write(str(out_wav_path), TARGET_SR, float32_to_int16(noisy_audio))

                # 如果复制原始的标注 txt
                if COPY_LABEL_TXT:
                    label_path = clean_path.with_suffix(".txt")
                    if label_path.exists():
                        shutil.copy2(label_path, out_wav_path.with_suffix(".txt"))

                # 记录到 CSV
                writer.writerow([str(out_wav_path), str(clean_path), desc, current_snr, TARGET_SR, len(clean)])

    print(f"完成！已生成 4 个信噪比级别的子文件夹，输出至 {OUT_DIR}")

if __name__ == "__main__":
    main()
