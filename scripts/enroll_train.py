#!/usr/bin/env python
"""Train one person's wake-word head from a handful of enrollment clips.

This is the productionized path oww-train's LORA.md prototyped by hand for Om: the
dashboard's Wake Word page records a few "Hey Robin" takes, `python -m robin.admin
wake-clips-export` writes them out, this script trains the head, and `python -m
robin.admin wake-model-ingest` activates it (robin/README.md §4c).

The code lives here in talkback; the DATA and training deps live in the oww-train repo
(--oww-root): the base model, the anchor features, the RIR/noise lists, and the
openWakeWord checkout. Run it with oww-train's env, not talkback's:

    ~/miniconda3/envs/oww-train/bin/python scripts/enroll_train.py CLIPS_DIR --out OUTDIR

CLIPS_DIR holds positives/*.wav (at least one) and optionally negatives/*.wav — the
person's ordinary speech. 16 kHz mono.

The training recipe is FROZEN (oww-train LORA.md config, LORA_RANK.md seed-averaging);
the one selected knob is alpha, chosen AFTER training by rescaling the merged delta:

  clips -> QC -> 12x augment+featurize (the same path the base head's own data took;
           its random EQ / band-stop / distortion / noise / RIR draws vary timbre and
           channel per copy)
        -> rank-4 zero-init LoRA on the Om-merged base, N seeds (always at the
           recipe's alpha = 2r = 8)
        -> weight-average the merged heads (single-seed recall spans 0.64-0.83; the
           average is what is stable)
        -> alpha sweep: rescale the merged delta to alpha in {1, 2, 4, 6, 8} and keep
           the candidate with the best AUC (peak scores of the person's takes ranked
           against held-out ACAV background, threshold-free). Ties go to the SMALLER
           alpha, so a weak adapter degrades toward the base instead of paying
           background drift for nothing. Selection uses ONLY the person's takes plus
           background -- never oww-train's samples/ benchmark, which stays report-only.
        -> threshold recalibrated to 0.2 FA/h on held-out ACAV background, for the
           chosen candidate
        -> merged export, unchanged ONNX contract -- a drop-in for the tablet

(An earlier +50% "voice variant" pre-pass — librosa pitch/tempo shifts of each take —
was removed: the phase-vocoder artifacts scored 0.001-0.023 on the wake head, i.e. the
variants entered training as noise labeled positive and reproduced exactly the poisoned-
positive failure they were meant to dilute.)

Anchors (ACAV100M background + v3 synthetic positives) are replayed in every batch;
without them a few dozen clips destroy background rejection (2/h -> ~1000/h).

The FA@0.5 gate decides `status` in the manifest: "failed_gate" means wake-model-ingest
refuses it and the tablet keeps the shared base. The threshold in the manifest is part
of the artifact -- serving the head without it is a regression (LORA.md).
"""
import argparse, datetime, hashlib, json, logging, sys, warnings
from pathlib import Path

import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

DEFAULT_OWW_ROOT = Path.home() / "Project/oww-train"
BASE = "lora/rank_sweep/hey_robin_om_r8_avg9.onnx"   # 9-seed average: same recall as the
BASE_VERSION = "v3+om_r8avg9"                        # shipped r=4, 3x lower FA variance
CONFIG = dict(r=4, wneg=1000, lr=1e-3, steps=150)    # frozen -- LORA.md's chosen config
AUC_EPS = 1e-3      # alpha-grid AUCs closer than this are a tie; the smaller alpha wins
TOTAL_LENGTH, ROUNDS = 32000, 12
DUR_BOUNDS = {"positives": (0.25, 4.0), "negatives": (0.15, 6.0)}

OWW = None   # oww-train repo root (Path), set in main()
L = None     # oww-train's lora_train module, imported in main() once sys.path is set


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def qc(files, lo, hi):
    """Format/duration/energy checks only. Content QC (did they actually say the wake
    word?) already ran at upload time (robin/api/wake.py's STT gate); this catches
    broken exports, not bad takes."""
    keep, rej = [], {}
    for f in files:
        try:
            info = sf.info(str(f))
        except Exception as e:                      # noqa: BLE001
            rej[f.name] = f"unreadable: {e}"; continue
        if info.samplerate != 16000 or info.channels != 1:
            rej[f.name] = f"not 16 kHz mono ({info.samplerate} Hz, {info.channels} ch)"; continue
        dur = info.frames / info.samplerate
        if not (lo <= dur <= hi):
            rej[f.name] = f"duration {dur:.2f}s outside [{lo}, {hi}]s"; continue
        if np.abs(sf.read(str(f))[0]).max() < 0.01:
            rej[f.name] = "near-silent"; continue
        keep.append(f)
    return keep, rej


def featurize(clips, out_dir, tag, rounds):
    """augment_clips -> compute_features: the identical path the base head's data took,
    with splits_v3's TRAIN rir/noise lists. Cached per output dir."""
    from openwakeword.data import augment_clips
    from openwakeword.utils import compute_features_from_generator
    out = out_dir / f"{tag}.npy"
    if out.exists():
        return np.load(out)
    rd = lambda f: [l.strip() for l in open(OWW / "splits_v3" / f) if l.strip()]
    lst = [str(c) for c in clips] * rounds
    gen = augment_clips(lst, total_length=TOTAL_LENGTH, batch_size=16,
                        background_clip_paths=rd("noise_train.txt"),
                        RIR_paths=rd("rirs_train.txt"))
    compute_features_from_generator(gen, n_total=len(lst), clip_duration=TOTAL_LENGTH,
                                    output_file=str(out), device="gpu", ncpu=1)
    a = np.load(out)
    print(f"  {tag}: {a.shape} from {len(clips)} clips x {rounds}")
    return a


def load_anchors(rng):
    """ACAV100M background + v3 synthetic positives, exactly as oww-train/lora_om.py
    drew them."""
    nm = json.load(open(OWW / "splits_v3/negatives.json"))
    trn = np.load(nm["train_slice"]["path"], mmap_mode="r")
    aneg = np.ascontiguousarray(trn[np.sort(rng.choice(len(trn), 150_000, replace=False))],
                                dtype=np.float32)
    spf = np.load(OWW / "my_custom_model_v3/hey_robin/positive_features_train.npy", mmap_mode="r")
    apos = np.ascontiguousarray(spf[np.sort(rng.choice(len(spf), 50_000, replace=False))],
                                dtype=np.float32)
    val = np.load(nm["validation_slice"]["path"], mmap_mode="r")
    s0 = nm["validation_slice"]["selection"]
    vf = np.ascontiguousarray(val[s0["start"]:s0["end"]], dtype=np.float32)
    sel = np.lib.stride_tricks.sliding_window_view(vf, (16, vf.shape[1]))[:, 0][:len(vf) - 16]
    return apos, aneg, sel, len(sel) / 12.5 / 3600


def fit(base_onnx, P, N, apos, aneg, seed):
    """One LoRA fit at the frozen config; returns the merged plain head."""
    import torch
    import torch.nn.functional as Fn
    g = torch.Generator().manual_seed(seed)
    net = L.apply_lora(L.load_onnx_weights(L.base_net(), base_onnx),
                       CONFIG["r"], 2 * CONFIG["r"]).to(L.DEV).train()
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=CONFIG["lr"])
    T = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32).to(L.DEV)
    Pt, AP, AN = T(P), T(apos), T(aneg)
    Nt = T(N) if N is not None else AN      # no real negatives: that slot draws anchors too
    d = lambda t, k: t[torch.randint(len(t), (k,), generator=g)]
    y = torch.cat([torch.ones(64), torch.zeros(544)]).to(L.DEV)
    w = torch.cat([torch.ones(64), torch.full((544,), float(CONFIG["wneg"]))]).to(L.DEV)
    for _ in range(CONFIG["steps"]):
        x = torch.cat([d(Pt, 32), d(AP, 32), d(Nt, 32), d(AN, 512)])
        opt.zero_grad()
        Fn.binary_cross_entropy(net(x)[:, 0].clamp(1e-6, 1 - 1e-6), y, weight=w).backward()
        opt.step()
    return L.merge_lora(net)


def average(nets):
    """Weight-average merged heads. All share the same frozen base, so this equals
    averaging the adapters -- the trick that stabilized r=8 in LORA_RANK.md."""
    import torch
    out = nets[0]
    sd = {k: torch.stack([n.state_dict()[k].float() for n in nets]).mean(0)
          for k in out.state_dict()}
    out.load_state_dict(sd)
    return out


def fa_and_thr(s, sel_h):
    """From background window scores: FA/h at the fixed 0.5 threshold (the gate) and
    the threshold that holds 0.2 FA/h (what ships next to the model)."""
    ths = np.unique(np.concatenate([np.linspace(0, 1, 2001), s]))
    return (float((s >= 0.5).sum() / sel_h),
            float(min(t for t in ths if (s >= t).sum() / sel_h <= 0.2 + 1e-12)))


def auc_of(peaks, s):
    """P(a take outscores a random background window) — the alpha-selection metric."""
    return float(np.mean([(s < p).mean() + 0.5 * (s == p).mean() for p in peaks]))


def clip_peaks(onnx_path, files):
    """Peak score per clip via predict_clip -- the protocol the tablet runs."""
    import openwakeword.model
    m = openwakeword.model.Model(wakeword_models=[str(onnx_path)], inference_framework="onnx")
    out = []
    for f in files:
        m.reset()
        out.append(max(list(x.values())[0] for x in m.predict_clip(str(f))))
    return np.array(out)


def main():
    global OWW, L
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("clips", type=Path, help="dir with positives/*.wav [negatives/*.wav]")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--person", default=None, help="label in the manifest (default: clips dir name)")
    ap.add_argument("--oww-root", type=Path, default=DEFAULT_OWW_ROOT,
                    help="oww-train repo (base model, anchors, openWakeWord checkout)")
    ap.add_argument("--base", default=BASE, help="base head ONNX, relative to --oww-root")
    ap.add_argument("--base-version", default=BASE_VERSION)
    ap.add_argument("--seeds", type=int, default=9)
    ap.add_argument("--fa-budget", type=float, default=2.0, help="gate: max FA/h at thr 0.5")
    a = ap.parse_args()

    OWW = a.oww_root.expanduser().resolve()
    if not (OWW / "splits_v3").is_dir():
        raise SystemExit(f"{OWW} does not look like the oww-train repo (no splits_v3/)")
    sys.path.insert(0, str(OWW))
    sys.path.insert(0, str(OWW / "openWakeWord"))
    import lora_train
    L = lora_train
    base_onnx = str(OWW / a.base)

    a.out.mkdir(parents=True, exist_ok=True)
    feat_dir = a.out / "features"; feat_dir.mkdir(exist_ok=True)

    clips, rejects = {}, {}
    for lab in ("positives", "negatives"):
        found = sorted((a.clips / lab).glob("*.wav"))
        clips[lab], rejects[lab] = qc(found, *DUR_BOUNDS[lab])
        for name, why in rejects[lab].items():
            print(f"  reject {lab}/{name}: {why}")
    if not clips["positives"]:
        raise SystemExit("no positives survived QC; nothing to train on")
    print(f"clips: {len(clips['positives'])} positives, {len(clips['negatives'])} negatives")

    # torch reconstruction of the base must agree with the ONNX the tablet would run
    rng = np.random.default_rng(0)
    apos, aneg, sel, sel_h = load_anchors(rng)
    bnet = L.load_onnx_weights(L.base_net(), base_onnx)
    import onnxruntime as ort
    s = ort.InferenceSession(base_onnx, providers=["CPUExecutionProvider"])
    probe = np.ascontiguousarray(apos[:64])
    o = np.array([s.run(None, {s.get_inputs()[0].name: probe[i:i + 1]})[0][0, 0]
                  for i in range(len(probe))])
    diff = float(np.abs(o - L.score(bnet, probe)).max())
    assert diff < 1e-4, f"torch base does not match {base_onnx} (max diff {diff:.2e})"

    P = featurize(clips["positives"], feat_dir, "positives_aug", ROUNDS)
    N = featurize(clips["negatives"], feat_dir, "negatives_aug", ROUNDS) \
        if clips["negatives"] else None

    trained_alpha = 2 * CONFIG["r"]
    print(f"training {a.seeds} seeds at frozen config {CONFIG} on {L.DEV} "
          f"(alpha={trained_alpha}) ...")
    net = average([fit(base_onnx, P, N, apos, aneg, seed) for seed in range(a.seeds)])

    # Test-time alpha grid: rescale the merged delta (trained at alpha=2r) to each
    # candidate and keep the best AUC; strict > means ties keep the SMALLER alpha, so a
    # weak adapter degrades toward the base rather than paying background drift for
    # nothing. AUC positives are the training takes (in-sample — a selection heuristic,
    # not an evaluation); negatives are the same held-out background the gate uses.
    base_sd = {k: v.cpu() for k, v in bnet.state_dict().items()}
    delta = {k: v.cpu() - base_sd[k] for k, v in net.state_dict().items()}
    sb = L.score(bnet, sel)
    base_fa, base_thr = fa_and_thr(sb, sel_h)
    base_peaks = clip_peaks(base_onnx, clips["positives"])
    print(f"\n{'alpha':>6} {'AUC':>8} {'FA@0.5':>8} {'thr@0.2FA':>10} {'recall':>7}")
    print(f"{'base':>6} {auc_of(base_peaks, sb):>8.4f} {base_fa:>8.2f} "
          f"{base_thr:>10.3f} {float((base_peaks >= base_thr).mean()):>7.2f}")
    cands, sweep = {}, []
    for alpha in (1, 2, 4, 6, 8):
        cand = L.base_net()
        cand.load_state_dict({k: base_sd[k] + (alpha / trained_alpha) * delta[k]
                              for k in base_sd})
        s = L.score(cand, sel)
        fa_c, thr_c = fa_and_thr(s, sel_h)
        cand_onnx = feat_dir / f"alpha_{alpha}.onnx"
        L.export_onnx(cand, cand_onnx)
        peaks = clip_peaks(cand_onnx, clips["positives"])
        row = {"alpha": alpha, "auc": auc_of(peaks, s), "fa_at_0.5_per_h": fa_c,
               "threshold": thr_c, "recall_insample": float((peaks >= thr_c).mean())}
        cands[alpha] = cand
        sweep.append(row)
        print(f"{alpha:>6} {row['auc']:>8.4f} {fa_c:>8.2f} {thr_c:>10.3f} "
              f"{row['recall_insample']:>7.2f}")
    # Selection: candidates that pass the FA budget first (all of them only if none
    # pass), then best AUC with a real tolerance — AUC saturates near 1.0 on small
    # enrollments, where 1e-4 differences are noise that once walked the pick to the
    # largest alpha for nothing. Within AUC_EPS the SMALLER alpha wins.
    pool = [r for r in sweep if r["fa_at_0.5_per_h"] <= a.fa_budget] or sweep
    chosen = pool[0]
    for row in pool[1:]:
        if row["auc"] > chosen["auc"] + AUC_EPS:
            chosen = row
    net = cands[chosen["alpha"]]
    print(f"chosen: alpha={chosen['alpha']} "
          f"({'passes' if chosen['fa_at_0.5_per_h'] <= a.fa_budget else 'over'} FA budget)")
    fa, thr, recall = (chosen["fa_at_0.5_per_h"], chosen["threshold"],
                       chosen["recall_insample"])
    status = "ok" if fa <= a.fa_budget else "failed_gate"
    onnx_path = a.out / "wake_model.onnx"
    L.export_onnx(net, onnx_path)

    onnx_bytes = onnx_path.read_bytes()
    manifest = {
        "person": a.person or a.clips.name,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "base": {"path": a.base, "version": a.base_version,
                 "sha256": sha(Path(base_onnx).read_bytes())},
        "config": {**CONFIG, "seeds": a.seeds, "trained_alpha": trained_alpha},
        "alpha": chosen["alpha"],           # test-time scale the shipped head uses
        "alpha_sweep": sweep,               # every candidate's AUC/FA/thr/recall
        "clips": {lab: {f.name: sha(f.read_bytes()) for f in clips[lab]}
                  for lab in ("positives", "negatives")},
        "rejected": rejects,
        "threshold": thr,
        "fa_at_0.5_per_h": fa,
        "base_fa_at_0.5_per_h": base_fa,
        "fa_budget_per_h": a.fa_budget,
        "status": status,
        "enroll_recall_insample": recall,   # on the training clips themselves -- a sanity
        "onnx": onnx_path.name,             # floor, not an evaluation
        "sha256": sha(onnx_bytes),
        "bytes": len(onnx_bytes),
    }
    json.dump(manifest, open(a.out / "manifest.json", "w"), indent=2)
    print(f"\n{status.upper()}: alpha={chosen['alpha']} (best AUC {chosen['auc']:.4f})  "
          f"FA@0.5={fa:.2f}/h (base {base_fa:.2f}, budget {a.fa_budget})  "
          f"thr@0.2FA={thr:.3f} (base {base_thr:.3f})  in-sample recall={recall:.2f}")
    print(f"-> {onnx_path}  ({len(onnx_bytes)} bytes)\n-> {a.out / 'manifest.json'}")
    if status == "failed_gate":
        print("gate failed: do NOT activate this head; the tablet keeps the shared base")
        sys.exit(3)


if __name__ == "__main__":
    main()
