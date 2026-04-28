import os
from pathlib import Path
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import resample_poly
from math import gcd

DATA_DIR = Path(r"D:\VAD\1.vad_learn\data_processing\QUT_Dataset")
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

def process_audio_file(file_path, 
                       target_sr=None, 
                       to_mono=False, 
                       segment_seconds=None, 
                       delete_original=False,
                       in_dir=None,
                       out_dir=None):
    """
    音频处理的二次封装函数，功能可自由组合:
    :param file_path: 音频文件路径 (Path对象或字符串)
    :param target_sr: 目标采样率 (如 16000)。如果为 None，则保持原采样率不转换。
    :param to_mono: 是否强制转换为单声道 (True/False)。
    :param segment_seconds: 切割时长(秒)。如果为 None，则不进行切割，直接保持原长。
    :param delete_original: 切割完成后是否删除原有的长音频文件。
    :param in_dir: 原始音频的主目录（用于计算相对路径结构）。
    :param out_dir: 输出的主目录。如果这里传入了有效目录，导出的文件将保持子目录结构输出到此目录下；若为 None，直接原地读写。
    """
    file_path = Path(file_path)
    try:
        sr, data = wavfile.read(str(file_path))
        data = audio_to_float32(data)
        
        # 1. 单声道转换
        if to_mono and data.ndim > 1:
            data = np.mean(data, axis=1)
            
        # 2. 降采样/重采样
        if target_sr is not None and sr != target_sr:
            g = gcd(sr, target_sr)
            data = resample_poly(data, target_sr // g, sr // g).astype(np.float32)
            sr = target_sr
            
        # 计算相对存储目录
        if out_dir is not None and in_dir is not None:
            rel_dir = file_path.parent.relative_to(in_dir)
            target_save_dir = Path(out_dir) / rel_dir
            target_save_dir.mkdir(parents=True, exist_ok=True)
        else:
            # 如果不指定输出目录，则原地保存
            target_save_dir = file_path.parent

        # 3. 音频切割保存 或 直接覆盖保存
        if segment_seconds is not None and segment_seconds > 0:
            segment_samples = int(segment_seconds * sr)
            total_samples = len(data)
            num_segments = int(np.ceil(total_samples / segment_samples))
            
            for i in range(num_segments):
                start = i * segment_samples
                end = min(total_samples, start + segment_samples)
                segment_data = data[start:end]
                
                if len(segment_data) == 0:
                    continue
                    
                out_name = f"{file_path.stem}_{i:04d}.wav"
                out_path = target_save_dir / out_name
                wavfile.write(str(out_path), sr, float32_to_int16(segment_data))
                
            print(f"[{file_path.name}] 成功切割为 {num_segments} 段 {segment_seconds}s 音频 -> {target_save_dir}")
            
            if delete_original and (out_dir is None or str(target_save_dir) == str(file_path.parent)):
                # 如果是原地切割，则删除原文件（禁止跨目录删除防止误删）
                file_path.unlink()
        else:
            # 如果不切割，整体保存
            out_path = target_save_dir / file_path.name
            wavfile.write(str(out_path), sr, float32_to_int16(data))
            print(f"[{file_path.name}] 成功处理(单声道或降采样) -> {out_path}")
            
        return True
    except Exception as e:
        print(f"[{file_path.name}] 处理失败: {e}")
        return False

def main():
    if not DATA_DIR.exists():
        print(f"目录不存在: {DATA_DIR}")
        return
        
    wav_files = list(DATA_DIR.rglob("*.wav"))
    
    # ---------------- 核心配置区 ----------------
    # 根据你的需求，开启/关闭相应的功能：
    # 如果音频“已经采样好（16k/单声道）”，只需要切割，
    # 那么将 target_sr=None, to_mono=False, segment_seconds=15 即可！
    
    CFG_TARGET_SR = None      # 设为 16000 进行降采样，设为 None 不改变采样率
    CFG_TO_MONO = False       # 设为 True 转单声道，设为 False 不改变声道数
    CFG_SEGMENT_SEC = 15      # 每段切片秒数，设为 None 则不切割
    CFG_DELETE_ORIGINAL = False # 处理完是否删除原音频
    
    # 【新增功能】输出目录配置：
    # 如果希望保持原目录结构，但导出到一个新地方，请填写 CFG_OUT_DIR (例如 Path(r"D:\VAD\1.vad_learn\data_processing\QUT_Dataset_15s"))
    # 如果仍希望原地生成（像之前那样），请设置为 None
    CFG_OUT_DIR = Path(r"D:\VAD\1.vad_learn\data_processing\QUT_Dataset_15s")
    # --------------------------------------------
    
    action_str = []
    if CFG_TO_MONO: action_str.append("转单声道")
    if CFG_TARGET_SR: action_str.append(f"重采样为 {CFG_TARGET_SR}Hz")
    if CFG_SEGMENT_SEC: action_str.append(f"切割为 {CFG_SEGMENT_SEC}s片段")
    if CFG_OUT_DIR: action_str.append(f"另存至 {CFG_OUT_DIR}")
    
    print(f"找到 {len(wav_files)} 个音频，准备执行: {' + '.join(action_str) if action_str else '无操作'} ...")
    
    success_count = 0
    for idx, path in enumerate(wav_files, 1):
        # 如果是原地切割，避免重复切割生成的带 _0000 后缀的音频文件
        if CFG_OUT_DIR is None and CFG_SEGMENT_SEC and path.stem[-4:].isdigit() and "_" in path.stem:
            continue
            
        if process_audio_file(
            path, 
            target_sr=CFG_TARGET_SR, 
            to_mono=CFG_TO_MONO, 
            segment_seconds=CFG_SEGMENT_SEC,
            delete_original=CFG_DELETE_ORIGINAL,
            in_dir=DATA_DIR,
            out_dir=CFG_OUT_DIR
        ):
            success_count += 1
            
    print(f"全部完成！成功处理: {success_count}/{len(wav_files)}")

if __name__ == '__main__':
    main()
