import json
import random
import shutil
from pathlib import Path


OUTPUT_DIR = Path(
    "/data/zhangzhenyu/Dataset/XX_voicebank"
)

NUM_PER_SPEAKER = 3

DATASETS = {
    "aishell1": {
        "wav": "/data/zhangzhenyu/Dataset/data_aishell/wav",
        "trans": "/data/zhangzhenyu/Dataset/data_aishell/transcript/aishell_transcript_v0.8.txt"
    },
    "aishell2": {
        "wav": "/data/zhangzhenyu/Dataset/AISHELL-2/iOS/data/wav",
        "trans": "/data/zhangzhenyu/Dataset/AISHELL-2/iOS/data/trans.txt"
    },
    "aishell3": {
        "wav": "/data/zhangzhenyu/Dataset/data_aishell3_16k/data",
        "trans": "/data/zhangzhenyu/Dataset/data_aishell3_16k/transcript/aishell_transcript_v0.8.txt"
    }
}


def load_trans(path):
    result = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            item = line.strip().split()
            if len(item) < 2:
                continue
            result[Path(item[0]).stem] = "".join(item[1:])
    return result


def process_dataset(name, cfg, wav_dir, trans, lines, meta):

    speakers = {}

    for wav in Path(cfg["wav"]).rglob("*.wav"):
        spk = wav.parent.name
        speakers.setdefault(spk, []).append(wav)


    for spk, files in speakers.items():

        files = [
            x for x in files
            if x.stem in trans and len(trans[x.stem]) > 1
        ]

        if len(files) < NUM_PER_SPEAKER:
            continue

        spk_id = f"{name}_{spk}"

        for idx, wav in enumerate(
            random.sample(files, NUM_PER_SPEAKER),
            1
        ):
            new_name = f"{spk_id}_{idx:03d}.wav"

            shutil.copy2(
                wav,
                wav_dir / new_name
            )

            lines.append(
                f"{new_name}|{spk_id}|{trans[wav.stem]}"
            )

        meta[spk_id] = {
            "source": name,
            "speaker": spk,
            "samples": NUM_PER_SPEAKER
        }


def main():

    random.seed(2025)

    wav_dir = OUTPUT_DIR / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    meta = {}

    for name, cfg in DATASETS.items():

        print("processing", name)

        trans = load_trans(cfg["trans"])

        process_dataset(
            name,
            cfg,
            wav_dir,
            trans,
            lines,
            meta
        )


    (OUTPUT_DIR / "trans.txt").write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    (OUTPUT_DIR / "speaker_info.json").write_text(
        json.dumps(
            meta,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        "speakers:",
        len(meta),
        "wav:",
        len(lines)
    )


if __name__ == "__main__":
    main()
