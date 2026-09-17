import csv
import hashlib
import math
import random
import shutil
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import resample_poly


# ============================================================
# 1. 配置
# ============================================================

CLEAN_DIR = Path(r"your_clean_kws_path")

# 推荐目录结构：
#
# NOISE_DIR/
# ├── subway/
# │   ├── subway_001.wav
# │   ├── subway_002.wav
# │   └── ...
# ├── road/
# │   ├── road_001.wav
# │   ├── road_002.wav
# │   └── ...
#
# 每条 noise 可以是约 5 秒
NOISE_DIR = Path(r"your_noise_path")

OUT_DIR = Path(r"your_output_path")

# KWS 测试 SNR
SNR_LIST = [-10, -5, 0, 5, 10]

TARGET_SR = 16000

RANDOM_SEED = 2026

# 每种噪声环境生成多少个独立 realization。
#
# = 1: 最简单的 benchmark。
# = 3 或 5: 如果测试集不大，推荐增加到 3~5，
#   可以降低某一段 noise 恰好特别简单/特别困难带来的偶然性。
#
# 无论设置多少，每个 realization 在所有 SNR 下都使用完全相同 noise。
NOISE_REALIZATIONS_PER_TYPE = 1

# active speech power
ACTIVE_FRAME_MS = 25
ACTIVE_HOP_MS = 10

# 40 dB 比较宽松，但对于 clean KWS 数据通常可以接受。
# 如果 clean 前后静音很多、环境非常干净，可以尝试 30 dB。
ACTIVE_TOP_DB = 40.0

# 噪声安全保护
# 近似静音 noise 直接拒绝
MIN_NOISE_POWER = 1e-10

# 如果 noise 本身录音电平极低，算出来需要放大 100 倍以上，则认为 noise 文件异常。
# 注意：不应该 clamp scale，因为 clamp 后就不再满足目标 SNR。
MAX_NOISE_SCALE = 100.0

MAX_NOISE_SAMPLE_TRIES = 30

# 输出
# KWS 通常使用 PCM16，因此默认输出 int16 WAV。
OUTPUT_PEAK = 0.999

COPY_LABEL_TXT = True

# 推荐保留。
# clean reference 会使用和所有 noisy condition 完全相同的 common gain，
# 因此可公平比较 clean accuracy vs subway@10dB vs subway@0dB ...
WRITE_CLEAN_REFERENCE = True


# ============================================================
# 2. WAV 工具
# ============================================================

def audio_to_float32(data):
    if data.dtype == np.int16:
        return data.astype(np.float32) / 32768.0
    if data.dtype == np.int32:
        return data.astype(np.float32) / 2147483648.0
    if data.dtype == np.uint8:
        return (data.astype(np.float32) - 128.0) / 128.0

    data = data.astype(np.float32)
    if data.size == 0:
        return data

    max_abs = float(np.max(np.abs(data)))
    # 某些 WAV 虽然读取为 float，实际数值不是标准 [-1, 1]
    if max_abs > 1.5:
        data = data / max_abs
    return data.astype(np.float32)


def float32_to_int16(data):
    data = np.clip(data, -1.0, 1.0)
    return np.round(data * 32767.0).astype(np.int16)


def read_wav_mono(path, target_sr=16000):
    sr, data = wavfile.read(str(path))
    if data.size == 0:
        raise ValueError(f"Empty audio: {path}")

    data = audio_to_float32(data)
    # 多声道 -> mono
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    if sr <= 0:
        raise ValueError(f"Invalid sample rate: {sr}")

    if sr != target_sr:
        g = math.gcd(sr, target_sr)
        data = resample_poly(data, target_sr // g, sr // g).astype(np.float32)
        sr = target_sr

    if len(data) == 0:
        raise ValueError(f"Audio empty after resampling: {path}")

    return sr, data.astype(np.float32)


# ============================================================
# 3. 功率计算
# ============================================================

def signal_power(x):
    if len(x) == 0:
        return 0.0

    x = np.asarray(x, dtype=np.float64)
    pwr = float(np.mean(x * x))
    if not math.isfinite(pwr):
        return 0.0
    return max(pwr, 0.0)


def get_frame_starts(length, frame_len, hop_len):
    if length <= frame_len:
        return [0]

    starts = list(range(0, length - frame_len + 1, hop_len))
    # 把最后一个完整 frame 也纳入，减少尾部遗漏。
    last_start = length - frame_len
    if len(starts) == 0 or starts[-1] != last_start:
        starts.append(last_start)
    return starts


def active_speech_power(x, sr=16000, frame_ms=25, hop_ms=10, top_db=40.0):
    if len(x) == 0:
        return 0.0

    frame_len = max(1, int(sr * frame_ms / 1000.0))
    hop_len = max(1, int(sr * hop_ms / 1000.0))

    if len(x) < frame_len:
        return signal_power(x)

    # 向量化：先按 get_frame_starts 的帧起点（含末尾补帧），
    # 再用滑动窗一次选出所有帧并批量求功率。
    # 与旧版逐帧 Python 循环在数值上完全一致。
    starts = get_frame_starts(len(x), frame_len, hop_len)
    frames = sliding_window_view(np.asarray(x, dtype=np.float64), frame_len)[starts]
    powers = np.mean(frames * frames, axis=1)

    if powers.size == 0:
        return signal_power(x)

    max_power = float(np.max(powers))
    if max_power <= 0:
        return 0.0

    threshold = max_power * 10.0 ** (-top_db / 10.0)
    active = powers[powers >= threshold]
    if active.size == 0:
        return signal_power(x)

    return float(np.mean(active))


# ============================================================
# 4. 稳定随机种子
# ============================================================

def make_deterministic_rng(relative_clean_path, noise_type, realization_id):
    """不依赖 Python 内置 hash()。

    同一个 clean / noise_type / realization 每次运行都得到同一个随机序列；
    即使程序处理文件的顺序发生变化，结果也不会跟着整体漂移。
    """
    key = (
        f"{RANDOM_SEED}|{relative_clean_path.as_posix()}"
        f"|{noise_type}|{realization_id}"
    )
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return random.Random(seed)


# ============================================================
# 5. Noise 分类
# ============================================================

# 预载噪声缓存：path(str) -> 已重采样并去直流的 float32 mono 数组。
# 噪声文件只会被解码+重采样一次，之后所有 clean 的噪声轨构造都复用内存中的数组。
_NOISE_CACHE = {}


def preload_noise_cache(noise_groups):
    """启动时一次性预载所有噪声到内存。

    避免每条 clean、每个 realization 都重新从磁盘解码
    （尤其当噪声素材不是 16k、需要重采样时开销会放大很多）。
    """
    for files in noise_groups.values():
        for path in files:
            key = str(path)
            if key in _NOISE_CACHE:
                continue
            try:
                _, audio = read_wav_mono(path, TARGET_SR)
                audio = remove_dc(audio)
            except Exception:
                continue
            _NOISE_CACHE[key] = audio

    print(f"Noise cache loaded: {len(_NOISE_CACHE)} files")


def get_cached_noise(path):
    """从缓存取噪声音频；未命中则现场读取、去直流并写入缓存。"""
    key = str(path)
    if key in _NOISE_CACHE:
        return _NOISE_CACHE[key]

    try:
        _, audio = read_wav_mono(path, TARGET_SR)
        audio = remove_dc(audio)
    except Exception:
        return None

    _NOISE_CACHE[key] = audio
    return audio


def discover_noise_groups(noise_root):
    """推荐：NOISE_DIR/subway/*.wav、NOISE_DIR/road/*.wav。

    第一层子目录作为 noise type；如果 WAV 直接放在 NOISE_DIR 根目录，
    则归到 "default"。
    """
    groups = {}
    for path in sorted(noise_root.rglob("*.wav")):
        rel = path.relative_to(noise_root)
        noise_type = rel.parts[0] if len(rel.parts) >= 2 else "default"
        groups.setdefault(noise_type, []).append(path)
    return groups


# ============================================================
# 6. 构造与 clean 完全等长的 noise
# ============================================================

def remove_dc(x):
    """去掉 noise 的直流偏置。对公路、地铁等环境噪声通常更合理。"""
    if len(x) == 0:
        return x
    return (x - float(np.mean(x))).astype(np.float32)


def build_noise_track_once(noise_files, target_len, rng):
    """构造恰好 target_len 的环境噪声。

    clean <= 5 秒：从某一条约 5 秒 noise 中随机截取。
    clean > 5 秒：连续选择同类别的多个 noise 文件拼接。

    注意：对 KWS benchmark，这里不做 FG event active-power scaling，
    因为 subway / road 属于持续环境噪声。
    """
    if target_len <= 0:
        raise ValueError("target_len must be > 0")
    if not noise_files:
        raise ValueError("Empty noise file list")

    output_parts = []
    metadata = []
    remaining = target_len
    safety_count = 0

    while remaining > 0:
        safety_count += 1
        if safety_count > 100:
            raise RuntimeError("Too many attempts while building noise track")

        path = rng.choice(noise_files)
        noise = get_cached_noise(path)
        if noise is None or len(noise) == 0:
            continue
        if signal_power(noise) < MIN_NOISE_POWER:
            continue

        # 如果剩余长度 <= 当前 noise，随机截取一个等长片段
        if remaining <= len(noise):
            max_start = len(noise) - remaining
            start = rng.randint(0, max_start) if max_start > 0 else 0
            seg = noise[start:start + remaining]
            output_parts.append(seg.astype(np.float32))
            metadata.append((path.name, start, remaining))
            remaining = 0
        else:
            # clean 比单条 5s noise 更长：使用整条 noise 并继续选下一条，
            # 避免简单循环同一条 5s 噪声产生强周期性。
            output_parts.append(noise.astype(np.float32))
            metadata.append((path.name, 0, len(noise)))
            remaining -= len(noise)

    track = np.concatenate(output_parts)
    track = track[:target_len]  # 理论安全保护
    if len(track) != target_len:
        raise RuntimeError("Noise length mismatch")
    return track.astype(np.float32), metadata


def build_valid_noise_track(noise_files, clean_active_power, target_len, snr_list, rng):
    """不仅要求 noise 有功率，还提前检查最低 SNR 条件下是否需要异常大的 scale。

    如果某段 noise 电平太低，重新随机选择。
    """
    for _ in range(MAX_NOISE_SAMPLE_TRIES):
        noise, info = build_noise_track_once(noise_files, target_len, rng)
        noise_pwr = signal_power(noise)
        if noise_pwr < MIN_NOISE_POWER:
            continue

        valid = True
        for snr_db in snr_list:
            target_noise_power = clean_active_power / (10.0 ** (snr_db / 10.0))
            scale = math.sqrt(target_noise_power / noise_pwr)
            if not math.isfinite(scale) or scale <= 0 or scale > MAX_NOISE_SCALE:
                valid = False
                break

        if valid:
            return noise, info, noise_pwr

    raise RuntimeError("Cannot find a valid noise segment after multiple attempts")


# ============================================================
# 7. SNR
# ============================================================

def calculate_noise_scale(clean_active_power, noise_power, snr_db):
    if clean_active_power <= 0:
        raise ValueError("Invalid clean active power")
    if noise_power < MIN_NOISE_POWER:
        raise ValueError("Noise power too small")

    target_noise_power = clean_active_power / (10.0 ** (snr_db / 10.0))
    scale = math.sqrt(target_noise_power / noise_power)

    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid scale: {scale}")
    if scale > MAX_NOISE_SCALE:
        raise ValueError(f"Noise scale too large: {scale:.3f}")
    return scale


def calculate_snr_db(signal_pwr, noise_pwr):
    if signal_pwr <= 0 or noise_pwr <= 0:
        return float("nan")
    return 10.0 * math.log10(signal_pwr / noise_pwr)


# ============================================================
# 8. 文件夹名
# ============================================================

def snr_folder_name(snr):
    # 兼容整数和浮点 SNR
    value = int(snr) if float(snr).is_integer() else snr
    if value < 0:
        return f"snr_m{abs(value)}dB"
    if value > 0:
        return f"snr_p{value}dB"
    return "snr_0dB"


# ============================================================
# 9. 路径安全
# ============================================================

def is_subpath(child, parent):
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def validate_paths():
    clean_root = CLEAN_DIR.resolve()
    noise_root = NOISE_DIR.resolve()
    out_root = OUT_DIR.resolve()

    if not CLEAN_DIR.exists():
        raise FileNotFoundError(f"CLEAN_DIR not found: {CLEAN_DIR}")
    if not NOISE_DIR.exists():
        raise FileNotFoundError(f"NOISE_DIR not found: {NOISE_DIR}")

    if clean_root == noise_root:
        raise ValueError("CLEAN_DIR 和 NOISE_DIR 不能相同。")
    if clean_root == out_root:
        raise ValueError("OUT_DIR 和 CLEAN_DIR 不能相同。")
    if noise_root == out_root:
        raise ValueError("OUT_DIR 和 NOISE_DIR 不能相同。")

    if is_subpath(out_root, clean_root):
        raise ValueError("OUT_DIR 不能位于 CLEAN_DIR 内部。")
    if is_subpath(out_root, noise_root):
        raise ValueError("OUT_DIR 不能位于 NOISE_DIR 内部。")


# ============================================================
# 10. 标签复制
# ============================================================

def copy_label_if_exists(clean_path, out_wav_path):
    if not COPY_LABEL_TXT:
        return
    label_path = clean_path.with_suffix(".txt")
    if not label_path.exists():
        return
    shutil.copy2(label_path, out_wav_path.with_suffix(".txt"))


# ============================================================
# 11. 主流程
# ============================================================

def generate_kws_snr_dataset():
    validate_paths()

    clean_files = sorted(CLEAN_DIR.rglob("*.wav"))
    if not clean_files:
        raise FileNotFoundError(f"No clean WAV in {CLEAN_DIR}")

    noise_groups = discover_noise_groups(NOISE_DIR)
    if not noise_groups:
        raise FileNotFoundError(f"No noise WAV in {NOISE_DIR}")

    preload_noise_cache(noise_groups)

    print(f"Clean files: {len(clean_files)}")
    print("Noise types:")
    for noise_type, files in noise_groups.items():
        print(f"  {noise_type}: {len(files)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata_path = OUT_DIR / "metadata_kws_snr.csv"

    with open(metadata_path, "w", newline="", encoding="utf-8-sig") as f_meta:
        writer = csv.writer(f_meta)
        writer.writerow([
            "output_wav", "clean_wav",
            "noise_type", "realization",
            "noise_files_and_segments",
            "target_snr_db",
            "measured_active_snr_db",   # 目标定义使用这个
            # 如果 clean 有大量静音，global_snr 通常比 active_snr 低。
            "measured_global_snr_db",
            "clean_active_power", "clean_global_power",
            "noise_power_before_scale", "noise_scale",
            "common_output_gain",       # 同一条 clean 的所有条件使用相同 gain
            "sample_rate", "num_samples", "duration_sec",
        ])

        for clean_idx, clean_path in enumerate(clean_files, start=1):
            print(f"[{clean_idx}/{len(clean_files)}] {clean_path}")

            try:
                _, clean = read_wav_mono(clean_path, TARGET_SR)
            except Exception as e:
                print(f"Skip clean: {e}")
                continue

            if len(clean) == 0:
                continue

            relative_path = clean_path.relative_to(CLEAN_DIR)
            clean_active_pwr = active_speech_power(
                clean,
                sr=TARGET_SR,
                frame_ms=ACTIVE_FRAME_MS,
                hop_ms=ACTIVE_HOP_MS,
                top_db=ACTIVE_TOP_DB,
            )
            clean_global_pwr = signal_power(clean)

            if clean_active_pwr <= 0:
                print("Skip: invalid clean active power")
                continue

            # 先构造所有 noise condition，才能计算一个全局 common gain，
            # 保证所有 noise type / SNR 下 clean speech 的幅度完全一致。
            conditions = []
            max_peak = float(np.max(np.abs(clean)))

            for noise_type, noise_files in noise_groups.items():
                for realization_id in range(NOISE_REALIZATIONS_PER_TYPE):
                    rng = make_deterministic_rng(relative_path, noise_type, realization_id)

                    try:
                        noise_track, noise_info, noise_power = build_valid_noise_track(
                            noise_files, clean_active_pwr, len(clean), SNR_LIST, rng,
                        )
                    except Exception as e:
                        print(f"Noise build failed: {noise_type}, rep={realization_id}, {e}")
                        continue

                    # 所有 SNR 都使用这一条完全相同的 noise_track
                    for snr_db in SNR_LIST:
                        scale = calculate_noise_scale(clean_active_pwr, noise_power, snr_db)
                        scaled_noise = (noise_track * scale).astype(np.float32)
                        mixture = (clean + scaled_noise).astype(np.float32)

                        peak = float(np.max(np.abs(mixture)))
                        max_peak = max(max_peak, peak)

                        scaled_noise_power = signal_power(scaled_noise)
                        active_snr = calculate_snr_db(clean_active_pwr, scaled_noise_power)
                        global_snr = calculate_snr_db(clean_global_pwr, scaled_noise_power)

                        conditions.append({
                            "noise_type": noise_type,
                            "realization": realization_id,
                            "noise_info": noise_info,
                            "snr_db": snr_db,
                            "noise_power": noise_power,
                            "scale": scale,
                            "active_snr": active_snr,
                            "global_snr": global_snr,
                            "mixture": mixture,
                        })

            if not conditions:
                print("No valid noise conditions, skip.")
                continue

            # 一个 clean 的所有条件只计算一个 common gain
            common_gain = (OUTPUT_PEAK / max_peak) if max_peak > OUTPUT_PEAK else 1.0

            # clean reference（与所有 noisy condition 使用相同的 common gain）
            if WRITE_CLEAN_REFERENCE:
                clean_ref = clean * common_gain
                clean_out = OUT_DIR / "clean_reference" / relative_path
                clean_out.parent.mkdir(parents=True, exist_ok=True)
                wavfile.write(str(clean_out), TARGET_SR, float32_to_int16(clean_ref))
                copy_label_if_exists(clean_path, clean_out)

            # 输出 noisy conditions
            for cond in conditions:
                noise_type = cond["noise_type"]
                realization_id = cond["realization"]
                snr_db = cond["snr_db"]
                output_audio = cond["mixture"] * common_gain

                out_path = (
                    OUT_DIR / noise_type / f"rep_{realization_id:02d}"
                    / snr_folder_name(snr_db) / relative_path
                )
                out_path.parent.mkdir(parents=True, exist_ok=True)
                wavfile.write(str(out_path), TARGET_SR, float32_to_int16(output_audio))
                copy_label_if_exists(clean_path, out_path)

                noise_info_string = "|".join(
                    f"{name}@start={start}@len={length}"
                    for name, start, length in cond["noise_info"]
                )

                writer.writerow([
                    str(out_path), str(clean_path),
                    noise_type, realization_id,
                    noise_info_string,
                    snr_db,
                    f"{cond['active_snr']:.6f}",
                    f"{cond['global_snr']:.6f}",
                    f"{clean_active_pwr:.12e}",
                    f"{clean_global_pwr:.12e}",
                    f"{cond['noise_power']:.12e}",
                    f"{cond['scale']:.12e}",
                    f"{common_gain:.12e}",
                    TARGET_SR,
                    len(clean),
                    f"{len(clean) / TARGET_SR:.6f}",
                ])

    print("\nFinished.")
    print(f"Output: {OUT_DIR}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    generate_kws_snr_dataset()
