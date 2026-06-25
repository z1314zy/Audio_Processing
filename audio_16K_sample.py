import os
from pathlib import Path
import torchaudio
import warnings
warnings.filterwarnings("ignore")

def batch_resample_to_16k(input_dir, output_dir, target_sr=16000, recursive=True):
    """
    批量将文件夹内所有音频重采样到目标采样率（默认16000Hz）
    输出统一为标准16bit PCM格式WAV文件，自动保持原目录结构

    参数:
        input_dir (str): 原始音频所在文件夹路径
        output_dir (str): 重采样后音频的输出文件夹路径（不会覆盖原文件）
        target_sr (int): 目标采样率，默认16000
        recursive (bool): 是否递归处理所有子文件夹，默认True
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 支持的音频格式
    audio_suffix = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}

    # 收集所有音频文件
    if recursive:
        audio_files = [f for f in input_dir.rglob('*') 
                       if f.is_file() and f.suffix.lower() in audio_suffix]
    else:
        audio_files = [f for f in input_dir.iterdir() 
                       if f.is_file() and f.suffix.lower() in audio_suffix]

    if not audio_files:
        print("未找到支持的音频文件，请检查路径或格式")
        return

    success = 0
    failed = 0

    for file_path in audio_files:
        try:
            # 计算输出路径，保持原目录结构
            rel_path = file_path.relative_to(input_dir)
            out_path = output_dir / rel_path.with_suffix('.wav')
            out_path.parent.mkdir(parents=True, exist_ok=True)

            # 加载原始音频
            waveform, orig_sr = torchaudio.load(str(file_path))

            # 原采样率已匹配则直接转格式保存
            if orig_sr == target_sr:
                torchaudio.save(
                    str(out_path), waveform, target_sr,
                    encoding="PCM_S", bits_per_sample=16
                )
                print(f"采样率匹配，直接转存: {rel_path}")
                success += 1
                continue

            # 执行重采样（sinc_interp_hann为语音场景高质量算法）
            resampler = torchaudio.transforms.Resample(
                orig_freq=orig_sr,
                new_freq=target_sr,
                resampling_method='sinc_interp_hann'
            )
            resampled_wave = resampler(waveform)

            # 保存为标准16bit PCM WAV
            torchaudio.save(
                str(out_path), resampled_wave, target_sr,
                encoding="PCM_S", bits_per_sample=16
            )

            print(f"重采样完成: {rel_path} | {orig_sr}Hz → {target_sr}Hz")
            success += 1

        except Exception as e:
            print(f"处理失败: {file_path.name}，原因: {str(e)}")
            failed += 1
            continue

    print(f"\n===== 批量处理完成 =====")
    print(f"成功: {success} 个 | 失败: {failed} 个")
    print(f"输出目录: {output_dir.resolve()}")


# ==================== 配置并运行 ====================
if __name__ == '__main__':
    # 按需修改以下路径
    INPUT_FOLDER = r"D:\5月工作内容\工作相关\CosyVoice2\save_1\002"       # 原始音频文件夹路径
    OUTPUT_FOLDER = r"D:\5月工作内容\工作相关\CosyVoice2\save_1\003"  # 重采样后输出文件夹（自动创建）
    
    batch_resample_to_16k(
        input_dir=INPUT_FOLDER,
        output_dir=OUTPUT_FOLDER,
        target_sr=16000,
        recursive=True  # 设为False则只处理当前目录，不遍历子文件夹
    )
