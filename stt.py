"""STT stage: Parakeet TDT 0.6B v3 via NeMo. Resamples input to 16 kHz mono."""
import argparse
import os
import time

import soundfile as sf
import torch
import torchaudio


def load_model():
    import nemo.collections.asr as nemo_asr
    model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    return model


def to_16k_mono(wav_path, tmp_path):
    data, sr = sf.read(wav_path, dtype="float32")   # (n,) or (n, ch)
    if data.ndim == 2:
        data = data.mean(axis=1)                     # mono
    if sr != 16000:
        data = torchaudio.functional.resample(torch.from_numpy(data), sr, 16000).numpy()
    sf.write(tmp_path, data, 16000)
    return tmp_path


def transcribe(model, wav_path, tmp_path):
    with torch.inference_mode():
        out = model.transcribe([to_16k_mono(wav_path, tmp_path)])
    hyp = out[0]
    return hyp.text if hasattr(hyp, "text") else str(hyp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav_path")
    ap.add_argument("--out", default="out/transcript.txt")
    args = ap.parse_args()

    outdir = os.path.dirname(args.out) or "."
    os.makedirs(outdir, exist_ok=True)

    t0 = time.time()
    model = load_model()
    load = time.time() - t0

    t1 = time.time()
    text = transcribe(model, args.wav_path, os.path.join(outdir, "_stt_16k.wav"))
    infer = time.time() - t1

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"TIMING stage=stt load={load:.2f} infer={infer:.2f}")


if __name__ == "__main__":
    main()
