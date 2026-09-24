#!/usr/bin/env python3

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import traceback
import wave

VOICEBANK_DIR = Path('/data/zhangzhenyu/Dataset/AISHELL_voicebank')
OUTPUT_DIR = Path('/data/zhangzhenyu/Dataset/wakeword_dataset')
MODEL_DIR = Path('pretrained_models/Fun-CosyVoice3-0.5B')
# FunASR 的 model 参数要的是「模型名 / 仓库 id」（或 funasr 认可的本地模型目录），
# 不是 modelscope 的缓存目录本身（缓存目录名形如 iic--SenseVoiceSmall，
# 传进去会被当成未注册的模型名报 "is not registered"）。
ASR_MODEL = '/home/zhangzhenyu/.cache/modelscope/models/iic--SenseVoiceSmall/snapshots/master'
RANDOM_SEED = 2025
WAKEWORDS = ['团宝团宝']
STYLE_CONFIG = {
    'normal': {'mode': 'zero', 'speed': 1.0, 'count': 5},
    'slow': {'mode': 'zero', 'speed': 0.8, 'count': 2},
    'fast': {'mode': 'zero', 'speed': 1.2, 'count': 2},
    'very_fast': {'mode': 'zero', 'speed': 1.35, 'count': 1},
    'soft': {'mode': 'emotion', 'speed': 0.9, 'count': 2,
             'prompt': '请用轻柔温和的声音说出这句话。'},
    'happy': {'mode': 'emotion', 'speed': 1.0, 'count': 1,
              'prompt': '请用开心带微笑的声音说出这句话。'},
    'urgent': {'mode': 'emotion', 'speed': 1.15, 'count': 1,
               'prompt': '请用稍微着急的语气呼喊。'},
    'tired': {'mode': 'emotion', 'speed': 0.9, 'count': 1,
              'prompt': '请用疲惫但清晰的声音说出这句话。'},
}
PREFIX = 'You are a helpful assistant.'
END = '<|endofprompt|>'


@dataclass
class Task:
    speaker: str
    reference: str
    text: str
    style: str
    output: str
    transcript: str
    mode: str
    speed: float
    instruction: str
    seed: int


# ================= 1. 数据与任务 =================

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def load_voicebank(root, transcript_file=None):
    info = json.loads((root / 'speaker_info.json').read_text(encoding='utf-8'))
    speaker_ids = list(info) if isinstance(info, dict) else [str(x) for x in info]

    transcripts = {}
    if transcript_file:
        transcripts = json.loads(transcript_file.read_text(encoding='utf-8'))

    ordered_ids = sorted(speaker_ids, key=lambda x: (-len(x), x))
    bank = {speaker: [] for speaker in sorted(speaker_ids)}

    for wav in sorted((root / 'wav').glob('*.wav')):
        speaker = next((s for s in ordered_ids if wav.name.startswith(s + '_')), None)
        if speaker is None:
            continue

        text = transcripts.get(wav.name, '')
        txt = wav.with_suffix('.txt')
        if not text and txt.is_file():
            text = txt.read_text(encoding='utf-8')

        bank[speaker].append((str(wav.resolve()), text.strip()))

    return {speaker: refs for speaker, refs in bank.items() if refs}


def resolve_mode(reference_mode, style_mode, transcript):
    if style_mode == 'emotion':
        return 'instruct'
    if reference_mode == 'auto':
        return 'zero' if transcript else 'cross'
    return reference_mode


def select_reference(refs, reference_mode, style_mode, rng):
    """优先选择带 transcript 的参考音频，减少 auto 模式退化到 cross。"""
    if style_mode != 'emotion' and reference_mode in ('auto', 'zero'):
        transcribed = [ref for ref in refs if ref[1]]
        if transcribed:
            return rng.choice(transcribed)
    return rng.choice(refs)


def build_tasks(args):
    bank = load_voicebank(args.voicebank_dir, args.transcripts)
    if args.limit_speakers:
        bank = dict(list(bank.items())[:args.limit_speakers])

    rng = random.Random(args.seed)
    tasks = []

    for speaker, refs in bank.items():
        words = WAKEWORDS if args.per_word else [None]
        for word_idx, word in enumerate(words):
            for style, cfg in STYLE_CONFIG.items():
                for idx in range(cfg['count']):
                    reference, transcript = select_reference(
                        refs, args.reference_mode, cfg['mode'], rng
                    )
                    text = word or rng.choice(WAKEWORDS)
                    mode = resolve_mode(args.reference_mode, cfg['mode'], transcript)
                    suffix = f'_w{word_idx}' if args.per_word else ''
                    output = args.output_dir / 'wav' / f'{speaker}{suffix}_{style}_{idx:03d}.wav'

                    tasks.append(asdict(Task(
                        speaker=speaker,
                        reference=reference,
                        text=text,
                        style=style,
                        output=str(output),
                        transcript=transcript,
                        mode=mode,
                        speed=cfg['speed'],
                        instruction=cfg.get('prompt', ''),
                        seed=rng.randrange(2**32),
                    )))

    return tasks, bank


# ================= 2. TTS =================

def infer(model, task):
    common = {
        'prompt_wav': task['reference'],
        'stream': False,
        'speed': task['speed'],
    }

    if task['mode'] == 'zero':
        return model.inference_zero_shot(
            tts_text=task['text'],
            prompt_text=PREFIX + END + task['transcript'],
            **common,
        )

    if task['mode'] == 'cross':
        return model.inference_cross_lingual(
            tts_text=PREFIX + END + task['text'],
            **common,
        )

    return model.inference_instruct2(
        tts_text=task['text'],
        instruct_text=PREFIX + ' ' + task['instruction'] + END,
        **common,
    )


def collect_audio(outputs, torch):
    return torch.cat([
        output['tts_speech'].detach().float().cpu()
        for output in outputs
    ], dim=-1)


def audio_metrics(audio, sample_rate):
    duration = audio.shape[-1] / sample_rate
    rms = audio.double().square().mean().sqrt().item()
    peak = audio.abs().max().item()
    return {'duration': duration, 'rms': rms, 'peak': peak}


def signal_passed(metrics, quality, reference=False):
    min_duration = quality['min_reference_duration'] if reference else quality['min_duration']
    return metrics['duration'] >= min_duration and metrics['rms'] >= quality['min_rms']


def set_seed(seed, torch, np):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def synthesize(model, task, torch, np, quality, content_round):
    for attempt in range(quality['max_attempts']):
        seed_index = content_round * quality['max_attempts'] + attempt
        seed = (task['seed'] + seed_index * 1000003) % (2**32)
        set_seed(seed, torch, np)

        with torch.inference_mode():
            speech = collect_audio(infer(model, task), torch)

        metrics = audio_metrics(speech, model.sample_rate)
        if not signal_passed(metrics, quality):
            print(
                f'{Path(task["output"]).name}: signal-attempt={attempt + 1} '
                f'duration={metrics["duration"]:.3f}s rms={metrics["rms"]:.6f}',
                flush=True,
            )
            continue

        gain = min(1.0, 0.99 / metrics['peak'])
        # speech 是在 inference_mode 内产生的 inference tensor，
        # 出了该上下文后禁止原地修改，只能写成非原地形式。
        speech = speech * gain
        return speech, {
            'attempt': attempt + 1,
            'content_round': content_round + 1,
            'actual_seed': seed,
            'gain': gain,
            'raw_metrics': metrics,
        }

    raise RuntimeError(f'{task["output"]}: 信号质量连续 {quality["max_attempts"]} 次未通过')


def save_pcm16(path, speech, sample_rate, np):
    pcm = np.rint(speech.numpy()[0] * 32768).astype('<i2')
    with wave.open(str(path), 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


# ================= 3. ASR =================

def normalize_text(text):
    return re.sub(r'[^\u4e00-\u9fffA-Za-z0-9]+', '', text or '').lower()


def build_asr_model(asr_cfg):
    from funasr import AutoModel as FunASRAutoModel

    kwargs = {
        'model': asr_cfg['model'],
        'device': 'cuda:0',
        'disable_update': True,
        'disable_pbar': True,
    }

    if asr_cfg['use_vad']:
        kwargs['vad_model'] = 'fsmn-vad'
        kwargs['vad_kwargs'] = {'max_single_segment_time': 30000}

    return FunASRAutoModel(**kwargs)


def recognize_batch(asr_model, candidates, asr_cfg, postprocess):
    kwargs = {
        'input': [str(candidate['tmp']) for candidate in candidates],
        'cache': {},
        'language': asr_cfg['language'],
        'use_itn': asr_cfg['use_itn'],
    }

    # 当前任务都是极短唤醒词。无 VAD 时用 batch_size 才是真正的文件批量大小；
    # 开启 VAD 时再使用 batch_size_s / merge_vad。
    if asr_cfg['use_vad']:
        kwargs.update(
            batch_size_s=asr_cfg['batch_size_s'],
            merge_vad=True,
            merge_length_s=15,
        )
    else:
        kwargs['batch_size'] = min(asr_cfg['batch_size'], len(candidates))

    results = asr_model.generate(**kwargs)
    if len(results) != len(candidates):
        raise RuntimeError(
            f'ASR 返回数量异常: candidates={len(candidates)}, results={len(results)}'
        )

    checks = []
    for candidate, result in zip(candidates, results):
        expected = candidate['task']['text']
        expected_norm = normalize_text(expected)
        raw_text = result.get('text', '') if isinstance(result, dict) else str(result)
        text = postprocess(raw_text)
        recognized_norm = normalize_text(text)

        if asr_cfg['match'] == 'contains':
            passed = expected_norm in recognized_norm
        else:
            passed = recognized_norm == expected_norm

        checks.append({
            'passed': passed,
            'expected': expected,
            'expected_normalized': expected_norm,
            'raw_text': raw_text,
            'text': text,
            'recognized_normalized': recognized_norm,
            'match': asr_cfg['match'],
        })

    return checks


# ================= 4. 候选音频与结果 =================

def metadata_path(path):
    path = Path(path)
    return path.parent.parent / 'metadata' / f'{path.stem}.json'


def valid_existing(task):
    path = Path(task['output'])
    meta = metadata_path(path)
    if not path.is_file() or not meta.is_file():
        return False

    try:
        record = json.loads(meta.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return False

    return (
        record.get('content_verified') is True
        and record.get('text') == task['text']
        and record.get('asr', {}).get('passed') is True
    )


def validate_reference(task, torchaudio, quality):
    audio, sample_rate = torchaudio.load(task['reference'])
    metrics = audio_metrics(audio, sample_rate)
    if not signal_passed(metrics, quality, reference=True):
        raise RuntimeError(
            f'{task["reference"]}: 参考音频质量不合格 '
            f'(duration={metrics["duration"]:.3f}s, rms={metrics["rms"]:.6g})'
        )


def ensure_reference_valid(task, torchaudio, quality, reference_status):
    """每个 worker 内缓存参考音频检查结果；切换 reference 后也会检查。"""
    reference = task['reference']
    status = reference_status.get(reference)
    if status is True:
        return
    if isinstance(status, str):
        raise RuntimeError(status)

    try:
        validate_reference(task, torchaudio, quality)
    except Exception as exc:
        message = f'{type(exc).__name__}: {exc}'
        reference_status[reference] = message
        raise RuntimeError(message) from exc

    reference_status[reference] = True


def make_candidate(tts_model, state, torch, np, quality):
    task = state['task']
    path = Path(task['output'])
    tmp = path.with_name(f'{path.stem}.{os.getpid()}.r{state["content_round"]}.tmp.wav')

    speech, synthesis = synthesize(
        tts_model,
        task,
        torch,
        np,
        quality,
        state['content_round'],
    )
    save_pcm16(tmp, speech, tts_model.sample_rate, np)

    return {
        'state': state,
        'task': task,
        'path': path,
        'tmp': tmp,
        'sample_rate': tts_model.sample_rate,
        'frames': speech.shape[-1],
        'duration': speech.shape[-1] / tts_model.sample_rate,
        'metrics': audio_metrics(speech, tts_model.sample_rate),
        'synthesis': synthesis,
    }


def save_verified_candidate(candidate, check, asr_cfg):
    state = candidate['state']
    task = candidate['task']
    path = candidate['path']

    record = dict(
        task,
        sample_rate=candidate['sample_rate'],
        frames=candidate['frames'],
        duration=candidate['duration'],
        content_verified=True,
        metrics=candidate['metrics'],
        synthesis=candidate['synthesis'],
        asr=dict(
            check,
            model=asr_cfg['model'],
            language=asr_cfg['language'],
            use_itn=asr_cfg['use_itn'],
            content_round=state['content_round'] + 1,
            prior_rejects=state['asr_history'],
        ),
    )

    os.replace(candidate['tmp'], path)
    write_json(metadata_path(path), record)


def reject_candidate(candidate, check):
    state = candidate['state']
    task = candidate['task']
    state['asr_history'].append({
        'round': state['content_round'] + 1,
        'raw_text': check['raw_text'],
        'text': check['text'],
        'recognized_normalized': check['recognized_normalized'],
        'expected_normalized': check['expected_normalized'],
        'seed': candidate['synthesis']['actual_seed'],
        'reference': task['reference'],
        'mode': task['mode'],
        'speed': task['speed'],
    })
    state['content_round'] += 1
    candidate['tmp'].unlink(missing_ok=True)


def bump_counter(mapping, key):
    mapping[key] = mapping.get(key, 0) + 1


def record_task_failure(report, task, stage, error, history=None):
    report['failed'] += 1
    bump_counter(report['failed_by_stage'], stage)
    bump_counter(report['failed_by_style'], task.get('style', 'unknown'))
    bump_counter(report['failed_by_mode'], task.get('mode', 'unknown'))

    item = {
        'output': task.get('output'),
        'speaker': task.get('speaker'),
        'style': task.get('style'),
        'mode': task.get('mode'),
        'reference': task.get('reference'),
        'stage': stage,
        'error': error,
    }
    if history is not None:
        item['asr_history'] = history
    report['errors'].append(item)


def choose_retry_reference(state, reference_bank, reference_mode, switch_after):
    """同 speaker 内切换 prompt，避免一直在同一个坏 reference 上换 seed。"""
    if switch_after <= 0 or state['content_round'] % switch_after != 0:
        return False

    task = state['task']
    refs = reference_bank.get(task['speaker'], [])
    if len(refs) <= 1:
        return False

    style_mode = STYLE_CONFIG[task['style']]['mode']
    pool = refs
    if style_mode != 'emotion' and reference_mode in ('auto', 'zero'):
        transcribed = [ref for ref in refs if ref[1]]
        if transcribed:
            pool = transcribed

    pool = [ref for ref in pool if ref[0] != task['reference']]
    if not pool:
        return False

    retry_rng = random.Random(
        (task['seed'] + state['content_round'] * 2654435761) % (2**32)
    )
    reference, transcript = retry_rng.choice(pool)
    task['reference'] = reference
    task['transcript'] = transcript
    task['mode'] = resolve_mode(reference_mode, style_mode, transcript)
    return True


# ================= 5. GPU Worker =================

def load_worker_models(config, rank):
    repo = Path(config['repo_dir'])
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / 'third_party' / 'Matcha-TTS'))

    import numpy as np
    import torch
    import torchaudio
    from cosyvoice.cli.cosyvoice import AutoModel as CosyVoiceAutoModel
    from funasr.utils.postprocess_utils import rich_transcription_postprocess

    torch.set_num_threads(config['threads'])
    torch.cuda.set_device(0)

    print(
        f'worker={rank}, visible GPU={os.environ.get("CUDA_VISIBLE_DEVICES")}: '
        '加载 CosyVoice...',
        flush=True,
    )
    tts_model = CosyVoiceAutoModel(model_dir=config['model_dir'])

    print(f'worker={rank}: 加载 SenseVoice ASR...', flush=True)
    asr_model = build_asr_model(config['asr'])
    print(f'worker={rank}: TTS + ASR 加载完成', flush=True)

    return np, torch, torchaudio, tts_model, asr_model, rich_transcription_postprocess


def prepare_states(tasks, report):
    states = []

    for task in tasks:
        if valid_existing(task):
            report['skipped'] += 1
            continue

        path = Path(task['output'])
        path.unlink(missing_ok=True)
        metadata_path(path).unlink(missing_ok=True)

        states.append({
            'task': task,
            'content_round': 0,
            'asr_history': [],
        })

    return states


def process_batch(
    states,
    tts_model,
    asr_model,
    postprocess,
    torch,
    np,
    torchaudio,
    quality,
    asr_cfg,
    config,
    reference_bank,
    reference_status,
    report,
    rank,
):
    pending = states

    while pending:
        candidates = []

        # 数据级异常只淘汰当前样本，不再让整个 worker 退出。
        for state in pending:
            task = state['task']
            try:
                ensure_reference_valid(task, torchaudio, quality, reference_status)
                candidates.append(
                    make_candidate(tts_model, state, torch, np, quality)
                )
            except Exception:
                detail = traceback.format_exc()
                record_task_failure(report, task, 'tts_or_reference', detail)
                print(
                    f'worker={rank} ❌ 样本失败 {Path(task["output"]).name}: '
                    f'TTS/参考音频异常',
                    flush=True,
                )

        if not candidates:
            break

        # ASR 模型本身/批处理运行时异常属于系统级错误，继续抛给 worker。
        try:
            checks = recognize_batch(asr_model, candidates, asr_cfg, postprocess)
        except Exception:
            for candidate in candidates:
                candidate['tmp'].unlink(missing_ok=True)
            raise

        next_pending = []

        for candidate, check in zip(candidates, checks):
            path = candidate['path']
            state = candidate['state']
            task = candidate['task']

            if check['passed']:
                save_verified_candidate(candidate, check, asr_cfg)
                report['success'] += 1
                print(
                    f'worker={rank} ✅ ASR通过 {path.name}: {check["text"]!r}',
                    flush=True,
                )
                continue

            report['asr_rejects'] += 1
            print(
                f'worker={rank} ⚠️ ASR不通过 {path.name}: '
                f'期望={check["expected_normalized"]!r}, '
                f'识别={check["recognized_normalized"]!r}',
                flush=True,
            )

            reject_candidate(candidate, check)
            if state['content_round'] < asr_cfg['content_max_attempts']:
                if config['retry']['switch_reference_on_retry']:
                    switched = choose_retry_reference(
                        state,
                        reference_bank,
                        config['reference_mode'],
                        config['retry']['switch_reference_after'],
                    )
                    if switched:
                        report['reference_switches'] += 1
                        print(
                            f'worker={rank} ↻ 切换参考音频 {path.name}: '
                            f'{Path(task["reference"]).name} mode={task["mode"]}',
                            flush=True,
                        )

                report['regenerated'] += 1
                next_pending.append(state)
            else:
                error = f'ASR 内容连续 {asr_cfg["content_max_attempts"]} 次未通过'
                record_task_failure(
                    report,
                    task,
                    'asr_content',
                    error,
                    history=state['asr_history'],
                )
                print(
                    f'worker={rank} ❌ 放弃 {path.name}: {error}',
                    flush=True,
                )

        pending = next_pending


def worker(plan_path, rank, world):
    plan_path = Path(plan_path)
    report = {
        'rank': rank,
        'assigned': 0,
        'success': 0,
        'skipped': 0,
        'failed': 0,
        'asr_rejects': 0,
        'regenerated': 0,
        'reference_switches': 0,
        'failed_by_stage': {},
        'failed_by_style': {},
        'failed_by_mode': {},
        'errors': [],
        'fatal_errors': [],
    }

    try:
        plan = json.loads(plan_path.read_text(encoding='utf-8'))
        config = plan['config']
        quality = config['quality']
        asr_cfg = config['asr']
        reference_bank = plan.get('reference_bank', {})

        np, torch, torchaudio, tts_model, asr_model, postprocess = load_worker_models(config, rank)

        selected = plan['tasks'][rank::world]
        report['assigned'] = len(selected)
        batch_tasks = asr_cfg['batch_tasks']

        print(
            f'worker={rank}, tasks={len(selected)}, batch_tasks={batch_tasks}, '
            f'content_attempts={asr_cfg["content_max_attempts"]}',
            flush=True,
        )

        reference_status = {}
        for batch_no, start in enumerate(range(0, len(selected), batch_tasks), 1):
            batch = selected[start:start + batch_tasks]
            states = prepare_states(batch, report)

            print(f'worker={rank} batch={batch_no}: 处理 {len(states)} 条', flush=True)
            process_batch(
                states,
                tts_model,
                asr_model,
                postprocess,
                torch,
                np,
                torchaudio,
                quality,
                asr_cfg,
                config,
                reference_bank,
                reference_status,
                report,
                rank,
            )

            completed = min(start + len(batch), len(selected))
            accounted = report['success'] + report['skipped'] + report['failed']
            print(
                f'worker={rank} batch={batch_no} 完成；progress={completed}/{len(selected)} '
                f'accounted={accounted}/{len(selected)} success={report["success"]} '
                f'skipped={report["skipped"]} rejects={report["asr_rejects"]} '
                f'failed={report["failed"]} switches={report["reference_switches"]}',
                flush=True,
            )
    except Exception:
        detail = traceback.format_exc()
        report['fatal_errors'].append(detail)
        print(f'worker={rank} 系统级异常退出:\n{detail}', flush=True)

    report['accounted'] = report['success'] + report['skipped'] + report['failed']
    report['unaccounted'] = max(0, report['assigned'] - report['accounted'])
    write_json(plan_path.parent / 'reports' / f'worker_{rank}.json', report)
    return 1 if report['fatal_errors'] else 0


# ================= 6. 参数与主调度 =================

def parse_args():
    parser = argparse.ArgumentParser(description='CosyVoice 唤醒词数据生成 + SenseVoice 校验')
    parser.add_argument('--voicebank-dir', type=Path, default=VOICEBANK_DIR)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--model-dir', type=Path, default=MODEL_DIR)
    parser.add_argument('--repo-dir', type=Path, default=Path.cwd())
    parser.add_argument('--transcripts', type=Path)
    parser.add_argument('--reference-mode', choices=['auto', 'zero', 'cross'], default='auto')
    parser.add_argument('--num-gpus', type=int, default=8)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    parser.add_argument('--limit-speakers', type=int, default=0)
    parser.add_argument('--per-word', action='store_true')
    parser.add_argument('--dry-run', action='store_true')

    parser.add_argument('--max-attempts', type=int, default=3)
    parser.add_argument('--min-duration', type=float, default=0.20)
    parser.add_argument('--min-reference-duration', type=float, default=0.50)
    parser.add_argument('--min-rms', type=float, default=1e-5)

    parser.add_argument('--batch-tasks', type=int, default=32)
    parser.add_argument('--content-max-attempts', type=int, default=5)
    parser.add_argument('--asr-model', default=ASR_MODEL)
    parser.add_argument('--asr-language', default='zh')
    parser.add_argument('--asr-batch-size', type=int, default=64)
    parser.add_argument('--asr-batch-size-s', type=int, default=300)
    parser.add_argument('--asr-match', choices=['exact', 'contains'], default='exact')
    parser.add_argument('--asr-use-itn', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--asr-use-vad', action='store_true')

    parser.add_argument(
        '--switch-reference-on-retry',
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument('--switch-reference-after', type=int, default=2)

    parser.add_argument('--worker', nargs=3, metavar=('PLAN', 'RANK', 'WORLD'), help=argparse.SUPPRESS)
    args = parser.parse_args()

    for key in ('voicebank_dir', 'output_dir', 'model_dir', 'repo_dir'):
        setattr(args, key, getattr(args, key).resolve())
    if args.transcripts:
        args.transcripts = args.transcripts.resolve()

    return args


def build_config(args):
    return {
        'repo_dir': str(args.repo_dir),
        'model_dir': str(args.model_dir),
        'threads': args.threads,
        'reference_mode': args.reference_mode,
        'quality': {
            'max_attempts': args.max_attempts,
            'min_duration': args.min_duration,
            'min_reference_duration': args.min_reference_duration,
            'min_rms': args.min_rms,
        },
        'retry': {
            'switch_reference_on_retry': args.switch_reference_on_retry,
            'switch_reference_after': args.switch_reference_after,
        },
        'asr': {
            'model': str(args.asr_model),
            'language': args.asr_language,
            'batch_tasks': args.batch_tasks,
            'content_max_attempts': args.content_max_attempts,
            'batch_size': args.asr_batch_size,
            'batch_size_s': args.asr_batch_size_s,
            'match': args.asr_match,
            'use_itn': args.asr_use_itn,
            'use_vad': args.asr_use_vad,
        },
    }


def prepare_output(output_dir):
    for name in ('wav', 'metadata', 'logs', 'reports'):
        (output_dir / name).mkdir(parents=True, exist_ok=True)

    for stale in (output_dir / 'wav').glob('*.tmp.wav'):
        stale.unlink()


def get_devices(num_gpus):
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible:
        return [x.strip() for x in visible.split(',') if x.strip()][:num_gpus]
    return [str(i) for i in range(num_gpus)]


def launch_workers(plan_path, num_gpus, task_count, threads):
    devices = get_devices(num_gpus)
    if not devices:
        raise SystemExit('CUDA_VISIBLE_DEVICES 未提供任何可用 GPU')
    world = min(len(devices), task_count)
    processes = []

    for rank in range(world):
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=devices[rank],
            OMP_NUM_THREADS=str(threads),
            PYTHONUNBUFFERED='1',
        )
        log_path = plan_path.parent / 'logs' / f'worker_{rank}.log'
        with open(log_path, 'w', encoding='utf-8') as log:
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()),
                 '--worker', str(plan_path), str(rank), str(world)],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        processes.append(process)

    print(f'{world} 个 worker 已启动；进度见 {plan_path.parent / "logs"}', flush=True)

    for process in processes:
        process.wait()

    return processes


def summarize(output_dir, processes, total_tasks):
    reports = []
    for rank, process in enumerate(processes):
        path = output_dir / 'reports' / f'worker_{rank}.json'
        if path.is_file():
            reports.append(json.loads(path.read_text(encoding='utf-8')))
            continue

        error = f'worker {rank} 未生成报告，退出码={process.returncode}'
        print(error, flush=True)
        reports.append({
            'rank': rank,
            'assigned': 0,
            'success': 0,
            'skipped': 0,
            'failed': 0,
            'accounted': 0,
            'unaccounted': 0,
            'asr_rejects': 0,
            'regenerated': 0,
            'reference_switches': 0,
            'failed_by_stage': {},
            'failed_by_style': {},
            'failed_by_mode': {},
            'errors': [],
            'fatal_errors': [error],
        })

    totals = {
        key: sum(report.get(key, 0) for report in reports)
        for key in (
            'success', 'skipped', 'failed', 'asr_rejects',
            'regenerated', 'reference_switches', 'unaccounted'
        )
    }
    fatal_workers = sum(bool(report.get('fatal_errors')) for report in reports)
    accounted = totals['success'] + totals['skipped'] + totals['failed']

    ok = (
        all(process.returncode == 0 for process in processes)
        and fatal_workers == 0
        and totals['failed'] == 0
        and totals['unaccounted'] == 0
        and accounted == total_tasks
    )

    summary = {
        'ok': ok,
        'total': total_tasks,
        'accounted': accounted,
        'fatal_workers': fatal_workers,
        **totals,
        'workers': reports,
    }
    write_json(output_dir / 'reports' / 'summary.json', summary)
    print(
        json.dumps(
            {k: summary[k] for k in (
                'ok', 'total', 'accounted', 'success', 'skipped', 'failed',
                'unaccounted', 'fatal_workers', 'asr_rejects',
                'regenerated', 'reference_switches'
            )},
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if ok else 1


def run(args):
    tasks, reference_bank = build_tasks(args)

    mode_counts = Counter(task['mode'] for task in tasks)
    style_counts = Counter(task['style'] for task in tasks)
    transcript_counts = Counter(
        'with_transcript' if task['transcript'] else 'without_transcript'
        for task in tasks
    )
    print(
        f'Total tasks: {len(tasks)}; modes={dict(mode_counts)}; '
        f'transcripts={dict(transcript_counts)}; styles={dict(style_counts)}',
        flush=True,
    )

    if mode_counts.get('cross', 0):
        print(
            f'注意: 仍有 {mode_counts["cross"]} 条任务使用 cross 模式；'
            '这些 speaker 没有可用 transcript 时无法自动切到 zero-shot。',
            flush=True,
        )

    if args.dry_run:
        return 0

    prepare_output(args.output_dir)
    plan = {
        'config': build_config(args),
        'reference_bank': reference_bank,
        'tasks': tasks,
    }
    plan_path = args.output_dir / 'tasks.json'
    write_json(plan_path, plan)

    processes = launch_workers(plan_path, args.num_gpus, len(tasks), args.threads)
    return summarize(args.output_dir, processes, len(tasks))


if __name__ == '__main__':
    options = parse_args()
    if options.worker:
        sys.exit(worker(options.worker[0], int(options.worker[1]), int(options.worker[2])))
    sys.exit(run(options))
