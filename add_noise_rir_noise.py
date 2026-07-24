import os
import random
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal
from tqdm import tqdm
import shutil
import librosa
from multiprocessing import Process
import argparse


EPS = np.finfo(float).eps
RANDOM_SEED = 1632
np.random.seed(RANDOM_SEED)

AUDIO_EXTENSIONS = ['wav', 'flac']


def add_pyreverb(clean_speech, rir):
    reverb_speech = signal.fftconvolve(clean_speech, rir, mode='full')
    # make reverb_speech same length as clean_speech
    reverb_speech = reverb_speech[: clean_speech.shape[0]]

    return reverb_speech


def normalize(audio, target_level=-25):
    ''' Normalize the signal to the target level'''
    rms = np.sqrt(np.mean(audio**2))  # (audio ** 2).mean ** 0.5
    scalar = 10 ** (target_level / 20) / (rms + EPS)
    audio = audio * scalar
    return audio, scalar


def is_clipped(audio, clipping_threshold=0.99):
    return np.any(np.abs(audio) > clipping_threshold)


def mk_mixture(s1, s2, snr, noisy_target_level, target_level=-25):
    '''
        s1: reverberant speech
        s2: noise with the same length as s1
    '''
    scalar_0 = 1 / (np.max(np.abs(s1)) + EPS)
    clean = s1 * scalar_0
    rev_clean, _ = normalize(clean, target_level)
    rmsclean = np.sqrt(np.mean(rev_clean**2))  # (rev_clean**2).mean()**0.5

    noise = s2 / (np.max(np.abs(s2)) + EPS)
    noise, _ = normalize(noise, target_level)
    rmsnoise = np.sqrt(np.mean(noise**2))  # (noise**2).mean()**0.5

    noise_scalar_alpha = rmsclean / (10**(snr / 20) * rmsnoise + EPS)
    noise_new_level = noise * noise_scalar_alpha

    # Mix the reverberant clean speech and noise
    noisy = rev_clean + noise_new_level
    rmsnoisy = np.sqrt(np.mean(noisy**2))  # (noisy**2).mean()**0.5

    noisy_scalar = 10 ** (noisy_target_level / 20) / (rmsnoisy + EPS)
    noisy_speech = noisy * noisy_scalar

    # Check for clipping
    clipping_threshold = 0.99
    if is_clipped(noisy_speech, clipping_threshold):
        noisy_speech_max_amp_level = (
            np.max(np.abs(noisy_speech)) / (clipping_threshold - EPS)
        )
        noisy_speech = noisy_speech / noisy_speech_max_amp_level

    return noisy_speech.astype(np.float32)


def load_audio(audio_path, fs, first_channel=False):
    audio, audio_fs = sf.read(audio_path, dtype='float32')

    if len(audio.shape) > 1:
        if first_channel:
            audio = audio[:, 0]
        else:
            audio = np.mean(audio, axis=1)

    if audio_fs != fs:
        audio = librosa.resample(
            audio,
            orig_sr=audio_fs,
            target_sr=fs,
        )

    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) == 0:
        raise ValueError(f'audio is empty: {audio_path}')
    if not np.all(np.isfinite(audio)):
        raise ValueError(f'audio contains NaN or Inf: {audio_path}')
    if np.max(np.abs(audio)) <= 0:
        raise ValueError(f'audio is silent: {audio_path}')

    return audio


def get_noise(noise_list, fs, target_samples):
    noise_segments = []
    segment_info = []
    current_samples = 0
    consecutive_errors = 0
    max_consecutive_errors = max(20, len(noise_list) * 2)

    while current_samples < target_samples:
        noise_idx = random.randint(0, len(noise_list) - 1)
        noise_path = noise_list[noise_idx]

        try:
            noise = load_audio(noise_path, fs)
        except Exception as e:
            print(f'Warning: invalid noise file {noise_path}: {e}')
            consecutive_errors += 1
            if consecutive_errors >= max_consecutive_errors:
                raise RuntimeError(
                    'Too many invalid noise files. Please check noise_dir.'
                )
            continue

        consecutive_errors = 0
        remaining_samples = target_samples - current_samples

        if len(noise) > remaining_samples:
            max_start = len(noise) - remaining_samples
            start = np.random.randint(0, max_start + 1)
            noise_segment = noise[start: start + remaining_samples]
        else:
            start = 0
            noise_segment = noise

        noise_segments.append(noise_segment)
        segment_info.append(
            f'{noise_path}[{start}:{start + len(noise_segment)}]'
        )
        current_samples += len(noise_segment)

    noise = np.concatenate(noise_segments)[:target_samples]
    return noise, '|'.join(segment_info)


def get_rir(rir_list, fs):
    rir_idx = np.random.randint(0, len(rir_list))
    rir_path = rir_list[rir_idx]
    rir = load_audio(rir_path, fs, first_channel=True)

    return rir, rir_path


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_clean_list(data_root):
    clean_list = []
    output_paths = set()
    audio_files = librosa.util.find_files(
        data_root,
        ext=AUDIO_EXTENSIONS,
        recurse=True,
    )

    for clean_path in audio_files:
        try:
            audio_info = sf.info(clean_path)
            duration_seconds = audio_info.frames / audio_info.samplerate
            if duration_seconds <= 0:
                raise ValueError('duration is 0')
        except Exception as e:
            print(f'Warning: invalid clean audio {clean_path}: {e}')
            continue

        relative_path = os.path.relpath(clean_path, data_root)
        output_relative_path = (
            os.path.splitext(relative_path)[0] + '.wav'
        )
        if output_relative_path in output_paths:
            raise ValueError(
                f'duplicate output path generated: {output_relative_path}'
            )

        output_paths.add(output_relative_path)
        clean_list.append(
            (clean_path, output_relative_path, duration_seconds)
        )

    if len(clean_list) == 0:
        raise ValueError(
            f'No valid clean audio found under data_root: {data_root}'
        )

    return clean_list


def limit_clean_list(clean_list, total_hours):
    if total_hours == 0:
        return clean_list
    if total_hours < 0:
        raise ValueError('total_hours must be greater than or equal to 0')

    target_seconds = total_hours * 3600
    available_seconds = sum(item[2] for item in clean_list)
    if target_seconds >= available_seconds:
        if target_seconds > available_seconds:
            print(
                f'Warning: requested {total_hours:.3f} hours, but only '
                f'{available_seconds / 3600:.3f} hours are available. '
                'Every clean file will be processed once.'
            )
        return clean_list

    random_generator = random.Random(RANDOM_SEED)
    shuffled_clean_list = clean_list.copy()
    random_generator.shuffle(shuffled_clean_list)

    selected_clean_list = []
    selected_seconds = 0
    for clean_item in shuffled_clean_list:
        if selected_seconds >= target_seconds:
            break
        selected_clean_list.append(clean_item)
        selected_seconds += clean_item[2]

    selected_clean_list.sort(key=lambda item: item[1])
    return selected_clean_list


def paths_overlap(path_a, path_b):
    path_a = os.path.realpath(path_a)
    path_b = os.path.realpath(path_b)
    common_path = os.path.commonpath([path_a, path_b])

    return common_path == path_a or common_path == path_b


def process_audio(
    i,
    clean_list,
    noise_list,
    rir_list,
    snr_list,
    target_level_list,
    save_root,
    fs,
):
    clean_path, output_relative_path, _ = clean_list[i]

    # Clean
    try:
        clean = load_audio(clean_path, fs)
    except Exception as e:
        print(f'\nError reading clean audio file {clean_path}: {e}\n')
        return None

    # Noise: crop one file or concatenate multiple files to match clean length
    try:
        noise, noise_segments = get_noise(
            noise_list,
            fs,
            len(clean),
        )
    except Exception as e:
        print(f'\nError preparing noise for {clean_path}: {e}\n')
        return None

    # RIR
    try:
        rir, rir_path = get_rir(rir_list, fs)
        max_index = np.argmax(np.abs(rir))
        rir = rir[max_index:]
    except Exception as e:
        print(f'\nError reading RIR for {clean_path}: {e}\n')
        return None

    rev_clean = add_pyreverb(clean, rir)

    snri = snr_list[i]
    target_level_i = target_level_list[i]
    noisy = mk_mixture(
        rev_clean,
        noise,
        snri,
        target_level_i,
    )

    output_path = os.path.join(save_root, output_relative_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, noisy, fs)

    return [
        output_relative_path,
        clean_path,
        noise_segments,
        rir_path,
        snri,
        target_level_i,
        len(noisy) / fs,
        len(noisy),
        fs,
    ]


def process_chunk(args):
    # 解开元组参数
    (
        chunk,
        clean_list,
        noise_list,
        rir_list,
        snr_list,
        target_level_list,
        save_root,
        fs,
        seed,
        id,
    ) = args
    np.random.seed(seed)
    random.seed(seed)

    start_idx, end_idx = chunk
    results = []
    for i in tqdm(range(start_idx, end_idx), desc=f'Process {id}'):
        try:
            result = process_audio(
                i,
                clean_list,
                noise_list,
                rir_list,
                snr_list,
                target_level_list,
                save_root,
                fs,
            )
            if result is not None:
                results.append(result)
        except Exception as e:
            print(f'Error processing audio at index {i}: {e}')

    info = pd.DataFrame(
        results,
        columns=[
            'output_file',
            'clean_file',
            'noise_segments',
            'rir_file',
            'snr',
            'target_level',
            'duration_seconds',
            'num_samples',
            'sample_rate',
        ],
    )
    csv_file_path = os.path.join(save_root, f'.INFO_worker_{id}.csv.tmp')
    info.to_csv(csv_file_path, index=None)

    return results


def main(args):
    fs = args.fs
    snr_min = args.snr_min
    snr_max = args.snr_max
    target_level_lower = args.target_level_lower
    target_level_upper = args.target_level_upper
    total_hours = args.total_hours
    save_root = os.path.abspath(args.save_root)
    data_root = os.path.abspath(args.data_root)
    noise_dir = os.path.abspath(args.noise_dir)
    rir_dir = os.path.abspath(args.rir_dir)
    num_processes = args.num_processes

    if fs <= 0:
        raise ValueError('fs must be greater than 0')
    if snr_min > snr_max:
        raise ValueError('snr_min must not be greater than snr_max')
    if target_level_lower >= target_level_upper:
        raise ValueError(
            'target_level_lower must be smaller than target_level_upper'
        )
    if not os.path.isdir(data_root):
        raise ValueError(f'data_root does not exist: {data_root}')
    if not os.path.isdir(noise_dir):
        raise ValueError(f'noise_dir does not exist: {noise_dir}')
    if not os.path.isdir(rir_dir):
        raise ValueError(f'rir_dir does not exist: {rir_dir}')

    for input_name, input_dir in [
        ('data_root', data_root),
        ('noise_dir', noise_dir),
        ('rir_dir', rir_dir),
    ]:
        if paths_overlap(save_root, input_dir):
            raise ValueError(
                f'save_root must not overlap with {input_name}: {input_dir}'
            )

    clean_list = get_clean_list(data_root)
    clean_list = limit_clean_list(clean_list, total_hours)
    noise_list = librosa.util.find_files(
        noise_dir,
        ext=AUDIO_EXTENSIONS,
        recurse=True,
    )
    rir_list = librosa.util.find_files(
        rir_dir,
        ext='wav',
        recurse=True,
    )

    if len(noise_list) == 0:
        raise ValueError(f'No noise audio found under noise_dir: {noise_dir}')
    if len(rir_list) == 0:
        raise ValueError(f'No RIR audio found under rir_dir: {rir_dir}')

    total_num = len(clean_list)
    actual_hours = sum(item[2] for item in clean_list) / 3600
    print(
        'total_hours = {:.3f}, total_num = {}'.format(
            actual_hours,
            total_num,
        )
    )
    print(f'noise_num = {len(noise_list)}, rir_num = {len(rir_list)}')

    if args.clean_output and os.path.exists(save_root):
        shutil.rmtree(save_root)
    os.makedirs(save_root, exist_ok=True)

    np.random.seed(RANDOM_SEED)
    snr_list = np.random.uniform(snr_min, snr_max, size=total_num)
    target_level_list = np.random.randint(
        target_level_lower,
        target_level_upper,
        size=total_num,
    )

    num_processes = max(1, min(num_processes, total_num))
    chunk_size = total_num // num_processes

    chunks = []
    for i in range(num_processes):
        start_idx = i * chunk_size
        end_idx = (
            start_idx + chunk_size
            if i < num_processes - 1
            else total_num
        )
        chunks.append((
            (start_idx, end_idx),
            clean_list,
            noise_list,
            rir_list,
            snr_list,
            target_level_list,
            save_root,
            fs,
            RANDOM_SEED + i,
            i,
        ))

    processes = []
    for chunk in chunks:
        p = Process(target=process_chunk, args=(chunk,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    failed_processes = [p.pid for p in processes if p.exitcode != 0]
    if failed_processes:
        raise RuntimeError(f'Worker processes failed: {failed_processes}')

    csv_files = [
        os.path.join(save_root, f'.INFO_worker_{i}.csv.tmp')
        for i in range(num_processes)
    ]
    dfs = []
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            dfs.append(df)

    info_columns = [
        'output_file',
        'clean_file',
        'noise_segments',
        'rir_file',
        'snr',
        'target_level',
        'duration_seconds',
        'num_samples',
        'sample_rate',
    ]
    if len(dfs) > 0:
        final_df = pd.concat(dfs, ignore_index=True)
    else:
        final_df = pd.DataFrame(columns=info_columns)

    final_csv_path = os.path.join(save_root, 'INFO.csv')
    final_df.to_csv(final_csv_path, index=None)

    # 删除子 CSV 文件
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            os.remove(csv_file)

    print(
        f'Finished: {len(final_df)}/{total_num} files were generated. '
        f'INFO: {final_csv_path}'
    )

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            'Add reverberation and noise to a general variable-length '
            'clean speech dataset.'
        )
    )
    parser.add_argument(
        '--fs',
        type=int,
        default=16000,
        help='Output sampling frequency',
    )
    parser.add_argument(
        '--snr_min',
        type=float,
        default=-10,
        help='Minimum value of SNR',
    )
    parser.add_argument(
        '--snr_max',
        type=float,
        default=15,
        help='Maximum value of SNR',
    )
    parser.add_argument(
        '--target_level_lower',
        type=int,
        default=-35,
        help='Minimum value of target level',
    )
    parser.add_argument(
        '--target_level_upper',
        type=int,
        default=-5,
        help='Maximum value of target level',
    )
    parser.add_argument(
        '--total_hours',
        type=float,
        default=0,
        help='Hours to generate; 0 means process every clean file once',
    )
    parser.add_argument(
        '--save_root',
        type=str,
        required=True,
        help='Root directory to save the generated dataset',
    )
    parser.add_argument(
        '--data_root',
        type=str,
        required=True,
        help='Root directory of the variable-length clean speech dataset',
    )
    parser.add_argument(
        '--noise_dir',
        type=str,
        required=True,
        help='Root directory of the noise dataset',
    )
    parser.add_argument(
        '--rir_dir',
        type=str,
        required=True,
        help='Directory of impulse responses',
    )
    parser.add_argument(
        '--clean_output',
        type=str2bool,
        default=False,
        help='Whether to clear save_root before generation',
    )
    parser.add_argument(
        '--num_processes',
        type=int,
        default=max(1, os.cpu_count() or 1),
        help='Number of processes to use for parallel processing',
    )

    args = parser.parse_args()
    main(args)
