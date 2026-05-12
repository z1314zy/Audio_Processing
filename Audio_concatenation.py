# -*- coding: utf-8 -*-

import os
import re
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


# =========================================================
# 配置区域
# =========================================================

INPUT_DIR = Path("your_path")
OUTPUT_DIR = Path("your_path")

TARGET_SECONDS = 10.0
MIN_REMAIN_SECONDS = 0.1

TARGET_SR = 16000
TO_MONO = True

# 是否递归读取子目录
RECURSIVE = False

# 最后一段不足 TARGET_SECONDS 是否保存
# True: 只要最后一段 >= MIN_REMAIN_SECONDS 就保存
# False: 最后一段不足 TARGET_SECONDS 直接丢弃
SAVE_LAST_INCOMPLETE = False

AUDIO_EXTS = [".wav", ".flac", ".ogg", ".mp3"]

OUTPUT_PREFIX = "concat_noise"

# WAV 输出格式
OUTPUT_SUBTYPE = "PCM_16"


def natural_key(text):
    """
    自然排序。
    例如：
    1.wav, 2.wav, 10.wav
    会按 1, 2, 10 排序。
    """
    return [
        int(s) if s.isdigit() else s.lower()
        for s in re.split(r"(\d+)", text)
    ]


def list_audio_files(input_dir, audio_exts, recursive=False):
    """
    获取音频文件列表，并按路径自然排序。
    """

    input_dir = Path(input_dir)

    if recursive:
        files = [
            p for p in input_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in audio_exts
        ]
    else:
        files = [
            p for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in audio_exts
        ]

    files.sort(key=lambda p: natural_key(str(p.relative_to(input_dir))))

    return files


def to_mono_audio(data):
    """
    转单通道。
    soundfile 读取后：
    单通道: shape = [samples]
    多通道: shape = [samples, channels]
    """

    if data.ndim == 1:
        return data

    return np.mean(data, axis=1)


def resample_audio(data, orig_sr, target_sr):
    """
    重采样到目标采样率。
    """

    if orig_sr == target_sr:
        return data

    g = math.gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g

    data = resample_poly(data, up, down)

    return data.astype(np.float32)


def read_audio(path, target_sr=16000, to_mono=True):
    """
    读取音频，并统一为：
    1. float32
    2. 指定采样率
    3. 单通道或保持原始通道
    """

    data, sr = sf.read(str(path), always_2d=False)

    data = data.astype(np.float32)

    if to_mono:
        data = to_mono_audio(data)

    if sr != target_sr:
        data = resample_audio(data, sr, target_sr)
        sr = target_sr

    data = np.asarray(data, dtype=np.float32)

    return data, sr


def save_audio(path, data, sr, subtype="PCM_16"):
    """
    保存音频。
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = np.asarray(data, dtype=np.float32)
    data = np.nan_to_num(data)
    data = np.clip(data, -1.0, 1.0)

    sf.write(str(path), data, sr, subtype=subtype)


def save_failed_log(failed_items, output_dir):
    """
    保存失败文件日志。
    """

    if not failed_items:
        return

    failed_path = Path(output_dir) / "failed_files.txt"

    with open(failed_path, "w", encoding="utf-8") as f:
        for path, reason in failed_items:
            f.write(f"{path}\t{reason}\n")

    print(f"[INFO] 失败文件列表已保存: {failed_path}")


def save_concat_map(map_items, output_dir):
    """
    保存拼接来源记录。
    """

    map_path = Path(output_dir) / "concat_map.tsv"

    with open(map_path, "w", encoding="utf-8") as f:
        f.write("output_file\tduration_s\tsource_files\n")

        for output_file, duration_s, source_files in map_items:
            source_text = " | ".join(source_files)
            f.write(f"{output_file}\t{duration_s:.6f}\t{source_text}\n")

    print(f"[INFO] 拼接来源记录已保存: {map_path}")


def concat_audio_by_target_length(
    input_dir,
    output_dir,
    target_seconds=10.0,
    min_remain_seconds=0.1,
    target_sr=16000,
    to_mono=True,
    recursive=False,
    save_last_incomplete=False,
    output_prefix="concat_noise",
    output_subtype="PCM_16",
):
    """
    按顺序拼接音频。

    规则：
    1. 按顺序读取音频；
    2. 当前拼接长度不足 target_seconds，就继续添加；
    3. 达到 target_seconds 后保存；
    4. 如果当前音频还有剩余：
       - 剩余 >= min_remain_seconds，继续作为下一条拼接音频的开头；
       - 剩余 < min_remain_seconds，直接丢弃；
    5. 最后一段不足 target_seconds 时，由 save_last_incomplete 控制是否保存。
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if target_seconds <= 0:
        raise ValueError("target_seconds 必须大于 0")

    if min_remain_seconds < 0:
        raise ValueError("min_remain_seconds 不能小于 0")

    target_samples = int(round(target_seconds * target_sr))
    min_remain_samples = int(round(min_remain_seconds * target_sr))

    audio_files = list_audio_files(
        input_dir=input_dir,
        audio_exts=AUDIO_EXTS,
        recursive=recursive
    )

    if len(audio_files) == 0:
        print(f"[ERROR] 输入目录中没有找到音频文件: {input_dir}")
        return

    print("=" * 80)
    print("音频拼接配置")
    print("=" * 80)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"音频数量: {len(audio_files)}")
    print(f"目标时长: {target_seconds:.3f}s")
    print(f"最小剩余保留时长: {min_remain_seconds:.3f}s")
    print(f"目标采样率: {target_sr}")
    print(f"是否递归读取: {recursive}")
    print(f"是否保存最后不足目标时长的片段: {save_last_incomplete}")
    print("=" * 80)

    output_index = 0

    current_parts = []
    current_len = 0
    current_source_files = []

    failed_items = []
    map_items = []

    total_input_duration = 0.0
    total_output_duration = 0.0
    total_dropped_duration = 0.0

    for audio_path in audio_files:

        try:
            audio, sr = read_audio(
                path=audio_path,
                target_sr=target_sr,
                to_mono=to_mono
            )
        except Exception as e:
            print(f"[ERROR] 读取失败: {audio_path}, reason={e}")
            failed_items.append((str(audio_path), repr(e)))
            continue

        if len(audio) == 0:
            print(f"[SKIP] 空音频: {audio_path}")
            failed_items.append((str(audio_path), "empty_audio"))
            continue

        audio_duration = len(audio) / float(target_sr)
        total_input_duration += audio_duration

        rel_name = str(audio_path.relative_to(input_dir))

        pos = 0
        audio_len = len(audio)

        while pos < audio_len:

            need_samples = target_samples - current_len
            remain_samples = audio_len - pos

            # 当前音频剩余部分足够填满当前输出
            if remain_samples >= need_samples:
                take_start = pos
                take_end = pos + need_samples

                current_parts.append(audio[take_start:take_end])
                current_len += need_samples

                current_source_files.append(
                    f"{rel_name}[{take_start / target_sr:.3f}-{take_end / target_sr:.3f}s]"
                )

                pos = take_end

                # 保存一条完整 target_seconds 的音频
                output_audio = np.concatenate(current_parts, axis=0)

                output_name = f"{output_prefix}_{output_index:06d}.wav"
                output_path = output_dir / output_name

                save_audio(
                    path=output_path,
                    data=output_audio,
                    sr=target_sr,
                    subtype=output_subtype
                )

                out_duration = len(output_audio) / float(target_sr)
                total_output_duration += out_duration

                map_items.append(
                    (output_name, out_duration, current_source_files.copy())
                )

                print(
                    f"[SAVE] {output_name}, "
                    f"duration={out_duration:.3f}s"
                )

                output_index += 1

                # 重置当前拼接缓存
                current_parts = []
                current_len = 0
                current_source_files = []

                # 当前音频切完 target 后，还剩多少
                remain_after_cut = audio_len - pos

                if 0 < remain_after_cut < min_remain_samples:
                    dropped_s = remain_after_cut / float(target_sr)
                    total_dropped_duration += dropped_s

                    print(
                        f"[DROP] {rel_name}, "
                        f"remain={dropped_s:.3f}s < "
                        f"{min_remain_seconds:.3f}s"
                    )

                    pos = audio_len

                # remain_after_cut >= min_remain_samples 时，不需要特殊处理
                # while 会继续处理剩余部分，它自然会成为下一条拼接音频的开头

            # 当前音频剩余部分不够填满当前输出，全部加入
            else:
                take_start = pos
                take_end = audio_len

                current_parts.append(audio[take_start:take_end])
                current_len += remain_samples

                current_source_files.append(
                    f"{rel_name}[{take_start / target_sr:.3f}-{take_end / target_sr:.3f}s]"
                )

                pos = audio_len

                print(
                    f"[ADD] {rel_name}, "
                    f"current={current_len / target_sr:.3f}s / "
                    f"{target_seconds:.3f}s"
                )

    # 处理最后一段不足 target_seconds 的尾巴
    if current_len > 0:
        last_duration = current_len / float(target_sr)

        if save_last_incomplete and current_len >= min_remain_samples:
            output_audio = np.concatenate(current_parts, axis=0)

            output_name = f"{output_prefix}_{output_index:06d}.wav"
            output_path = output_dir / output_name

            save_audio(
                path=output_path,
                data=output_audio,
                sr=target_sr,
                subtype=output_subtype
            )

            total_output_duration += last_duration

            map_items.append(
                (output_name, last_duration, current_source_files.copy())
            )

            print(
                f"[SAVE-LAST] {output_name}, "
                f"duration={last_duration:.3f}s"
            )

            output_index += 1

        else:
            total_dropped_duration += last_duration

            print(
                f"[DROP-LAST] 最后一段 duration={last_duration:.3f}s，"
                f"未保存"
            )

    save_failed_log(failed_items, output_dir)
    save_concat_map(map_items, output_dir)

    print("\n" + "=" * 80)
    print("拼接完成")
    print("=" * 80)
    print(f"输入音频数量: {len(audio_files)}")
    print(f"成功生成音频数量: {output_index}")
    print(f"读取失败数量: {len(failed_items)}")
    print(f"输入总时长: {total_input_duration:.3f}s")
    print(f"输出总时长: {total_output_duration:.3f}s")
    print(f"丢弃总时长: {total_dropped_duration:.3f}s")
    print(f"输出目录: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    concat_audio_by_target_length(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        target_seconds=TARGET_SECONDS,
        min_remain_seconds=MIN_REMAIN_SECONDS,
        target_sr=TARGET_SR,
        to_mono=TO_MONO,
        recursive=RECURSIVE,
        save_last_incomplete=SAVE_LAST_INCOMPLETE,
        output_prefix=OUTPUT_PREFIX,
        output_subtype=OUTPUT_SUBTYPE,
    )
