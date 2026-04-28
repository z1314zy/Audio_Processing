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
# 1. 路径配置
# =========================

CLEAN_DIR = Path(r"D:\VAD\1.vad_learn\data_processing\train_clean_100")
NOISE_DIR = Path(r"D:\VAD\1.vad_learn\data_processing\noise\station-qut")
OUT_DIR = Path(r"D:\VAD\1.vad_learn\data_processing\train_clean_100_station-qut")

SNR_LIST = [-10, -5, 0, 5, 10]

TARGET_SR = 16000
RANDOM_SEED = 2026

# True：同一条 clean 在不同 SNR 下使用同一段噪声，便于公平比较
# False：每个 SNR 都重新随机选择噪声，数据多样性更强
USE_SAME_NOISE_FOR_ALL_SNRS = True

# True：最后输出音频如果幅度超过 1，会整体归一化，避免削波
NORMALIZE_OUTPUT = True

# 如果 clean 同名有 .txt 标签文件，则复制到带噪数据目录
COPY_LABEL_TXT = True


# =========================
# 2. 基础音频工具
# =========================

def audio_to_float32(data):
    """
    将 wavfile.read 读出的数据转换为 float32，范围大致为 [-1, 1]
    """
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
        elif max_abs < 1e-10:
            # 处理全是 0 或极小值的情况
            pass
        return data


def float32_to_int16(data):
    """
    将 float32 音频保存为 int16
    """
    data = np.clip(data, -1.0, 1.0)
    return (data * 32767.0).astype(np.int16)


def read_wav_mono(path, target_sr=16000):
    """
    读取 wav，转单声道，必要时重采样到 target_sr
    """
    sr, data = wavfile.read(str(path))
    
    if data.size == 0:
        raise ValueError(f"Empty audio file: {path}")
    
    data = audio_to_float32(data)

    # 多声道转单声道
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # 重采样
    if sr != target_sr:
        g = gcd(sr, target_sr)
        up = target_sr // g
        down = sr // g
        data = resample_poly(data, up, down).astype(np.float32)
        sr = target_sr

    if len(data) == 0:
        raise ValueError(f"Audio is empty after processing: {path}")

    return sr, data.astype(np.float32)


def signal_power(x, eps=1e-12):
    """
    计算平均功率
    """
    if len(x) == 0:
        return eps
    return max(float(np.mean(x ** 2)), eps)


def active_speech_power(x, sr=16000, frame_ms=25, hop_ms=10, top_db=40, eps=1e-12):
    """
    估计纯净语音的有效语音功率。

    为什么不用整段功率？
    如果 clean 中有大量静音，用整段算功率会导致 SNR 不准。
    这里用简单能量门限，只统计较活跃的语音帧。
    """
    frame_len = int(sr * frame_ms / 1000)
    hop_len = int(sr * hop_ms / 1000)

    if len(x) < frame_len:
        return signal_power(x, eps)

    frame_powers = []
    for start in range(0, len(x) - frame_len + 1, hop_len):
        frame = x[start:start + frame_len]
        frame_powers.append(np.mean(frame ** 2))

    frame_powers = np.asarray(frame_powers, dtype=np.float32)

    if len(frame_powers) == 0:
        return signal_power(x, eps)

    max_power = np.max(frame_powers)

    if max_power <= eps:
        return signal_power(x, eps)

    # top_db = 40 表示保留能量不低于最大帧能量 40 dB 的帧
    threshold = max_power * (10 ** (-top_db / 10.0))
    active = frame_powers[frame_powers >= threshold]

    if len(active) == 0:
        return signal_power(x, eps)

    return max(float(np.mean(active)), eps)


# =========================
# 3. 噪声长度匹配
# =========================

def match_noise_length(noise, target_len, rng):
    """
    将噪声处理成与 clean 一样长。

    情况 1：噪声更长 → 随机裁剪
    情况 2：噪声更短 → 循环重复后截取
    """
    noise_len = len(noise)

    if noise_len == 0:
        raise ValueError("Empty noise audio.")

    if target_len <= 0:
        raise ValueError(f"Invalid target_len: {target_len}")

    if noise_len >= target_len:
        max_start = noise_len - target_len
        start = rng.randint(0, max_start) if max_start > 0 else 0
        noise_seg = noise[start:start + target_len]
        repeat_flag = 0
    else:
        # 从噪声随机位置开始循环取样
        start = rng.randint(0, noise_len - 1) if noise_len > 1 else 0
        indices = (np.arange(target_len) + start) % noise_len
        noise_seg = noise[indices]
        repeat_flag = 1

    return noise_seg.astype(np.float32), start, repeat_flag


def sample_noise_segment(noise_files, target_len, rng, target_sr=16000, max_tries=50):
    """
    随机选择一条有效噪声，并生成与 clean 等长的噪声片段
    """
    if target_len <= 0:
        raise ValueError(f"Invalid target_len: {target_len}")
    
    if len(noise_files) == 0:
        raise ValueError("No noise files available")

    for attempt in range(max_tries):
        noise_path = rng.choice(noise_files)

        try:
            sr, noise = read_wav_mono(noise_path, target_sr=target_sr)
        except Exception as e:
            print(f"Warning: failed to read noise {noise_path} (attempt {attempt+1}/{max_tries}), error: {e}")
            continue

        if len(noise) == 0:
            continue

        try:
            noise_seg, noise_start, repeat_flag = match_noise_length(
                noise,
                target_len,
                rng
            )
        except Exception as e:
            print(f"Warning: failed to match noise length for {noise_path}, error: {e}")
            continue

        if signal_power(noise_seg) > 1e-10:
            return noise_seg, noise_path, noise_start, repeat_flag

    raise RuntimeError(f"Failed to sample a valid noise segment after {max_tries} attempts.")


# =========================
# 4. 按目标 SNR 混合
# =========================

def mix_with_snr(clean, noise, snr_db, sr=16000):
    """
    按指定 SNR 混合 clean 和 noise。

    SNR = 10 * log10(P_clean / P_noise)

    noisy = clean + scale * noise
    """
    clean_power = active_speech_power(clean, sr=sr)
    noise_power = signal_power(noise)

    target_noise_power = clean_power / (10 ** (snr_db / 10.0))
    scale = math.sqrt(target_noise_power / noise_power)

    scaled_noise = noise * scale
    noisy = clean + scaled_noise

    peak = float(np.max(np.abs(noisy))) if len(noisy) > 0 else 0.0

    if NORMALIZE_OUTPUT and peak > 0.999:
        noisy = noisy / peak * 0.999

    return noisy.astype(np.float32), scale, clean_power, noise_power


def snr_folder_name(snr):
    return f"snr_{snr}dB"


# =========================
# 5. 主处理函数
# =========================

def generate_noisy_dataset():
    rng = random.Random(RANDOM_SEED)

    clean_files = sorted(CLEAN_DIR.rglob("*.wav"))
    noise_files = sorted(NOISE_DIR.rglob("*.wav"))

    if len(clean_files) == 0:
        raise FileNotFoundError(f"No clean wav files found in: {CLEAN_DIR}")

    if len(noise_files) == 0:
        raise FileNotFoundError(f"No noise wav files found in: {NOISE_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata_path = OUT_DIR / "metadata.csv"

    processed_count = 0
    skipped_count = 0
    error_count = 0

    try:
        with open(metadata_path, "w", newline="", encoding="utf-8-sig") as f_meta:
            writer = csv.writer(f_meta)
            writer.writerow([
                "output_wav",
                "clean_wav",
                "noise_wav",
                "snr_db",
                "sample_rate",
                "num_samples",
                "noise_start_sample",
                "noise_repeated",
                "noise_scale",
                "clean_power",
                "noise_power_before_scale",
            ])

            for idx, clean_path in enumerate(clean_files, start=1):
                print(f"[{idx}/{len(clean_files)}] Processing clean: {clean_path}")

                try:
                    sr, clean = read_wav_mono(clean_path, target_sr=TARGET_SR)
                except Exception as e:
                    print(f"Warning: failed to read clean {clean_path}, error: {e}")
                    error_count += 1
                    continue

                if len(clean) == 0:
                    print(f"Warning: empty clean audio, skip: {clean_path}")
                    skipped_count += 1
                    continue

                rel_path = clean_path.relative_to(CLEAN_DIR)

                # 同一条 clean 在不同 SNR 下使用同一段噪声
                noise_seg = None
                noise_path = None
                noise_start = None
                noise_repeated = None

                if USE_SAME_NOISE_FOR_ALL_SNRS:
                    try:
                        noise_seg, noise_path, noise_start, noise_repeated = sample_noise_segment(
                            noise_files,
                            target_len=len(clean),
                            rng=rng,
                            target_sr=TARGET_SR
                        )
                    except Exception as e:
                        print(f"Warning: failed to sample noise for {clean_path}, error: {e}")
                        error_count += 1
                        continue

                for snr in SNR_LIST:
                    if not USE_SAME_NOISE_FOR_ALL_SNRS:
                        try:
                            noise_seg, noise_path, noise_start, noise_repeated = sample_noise_segment(
                                noise_files,
                                target_len=len(clean),
                                rng=rng,
                                target_sr=TARGET_SR
                            )
                        except Exception as e:
                            print(f"Warning: failed to sample noise for {clean_path} at SNR {snr}dB, error: {e}")
                            error_count += 1
                            continue

                    try:
                        noisy, scale, clean_power, noise_power = mix_with_snr(
                            clean,
                            noise_seg,
                            snr_db=snr,
                            sr=TARGET_SR
                        )

                        out_subdir = OUT_DIR / snr_folder_name(snr) / rel_path.parent
                        out_subdir.mkdir(parents=True, exist_ok=True)

                        out_wav_path = out_subdir / rel_path.name

                        wavfile.write(
                            str(out_wav_path),
                            TARGET_SR,
                            float32_to_int16(noisy)
                        )

                        # 如果 clean 同名有 txt 标签，复制到对应目录
                        if COPY_LABEL_TXT:
                            label_path = clean_path.with_suffix(".txt")
                            if label_path.exists():
                                try:
                                    out_label_path = out_wav_path.with_suffix(".txt")
                                    shutil.copy2(label_path, out_label_path)
                                except Exception as e:
                                    print(f"Warning: failed to copy label file {label_path}, error: {e}")

                        writer.writerow([
                            str(out_wav_path),
                            str(clean_path),
                            str(noise_path),
                            snr,
                            TARGET_SR,
                            len(clean),
                            noise_start,
                            noise_repeated,
                            scale,
                            clean_power,
                            noise_power,
                        ])

                        processed_count += 1

                    except Exception as e:
                        print(f"Error: failed to process {clean_path} at SNR {snr}dB, error: {e}")
                        error_count += 1
                        continue

    except IOError as e:
        print(f"Error: failed to write metadata file {metadata_path}, error: {e}")
        raise

    print("\n" + "="*50)
    print(f"Processing complete!")
    print(f"Total processed samples: {processed_count}")
    print(f"Skipped samples: {skipped_count}")
    print(f"Error samples: {error_count}")
    print(f"Noisy dataset saved to: {OUT_DIR}")
    print(f"Metadata saved to: {metadata_path}")
    print("="*50)


if __name__ == "__main__":
    generate_noisy_dataset()