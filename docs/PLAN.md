# NeuralPlay — starting plan

**Deadline:** December 2026 (AIE623)  
**Goal:** one playable neural engine for **one** 2D side-scrolling Atari level. At play time the original engine is gone.

Adapted from the local draft `~/Desktop/hacettepe-studies/AI/NEURALPLAY_PLAN.md` (2 Nov 2026). Calendar weeks below follow that draft; if a week is already past, treat unfinished items as the next actions.

## Hardware split

| Machine | Role |
| --- | --- |
| Local **RX 7800 XT + Ubuntu 24.04** | Setup, data, VAE, play loop, evaluation |
| **EVREN H200** | Heavy dynamics training only |

Do **not** start a second game, a diffusion model, or a multi-game experiment.

---

## Milestone overview

| Week | Theme | Done when |
| --- | --- | --- |
| 1 | Machine + empty loop | ROCm/PyTorch see the GPU; dummy `play.py` reacts to keys |
| 2 | Data + game choice | One game chosen; train/test split; can plot frame + action |
| 3 | Visual model | Decoder rebuilds held-out frames; player/level recognisable |
| 4 | Dynamics + first playable | 5–10 s play without original engine |
| 5 | Stability | 32-step open-loop still looks like the game; intentional control |
| 6 | Report + freeze | Demo video, metrics table, short report; stop training |

---

## Week 1 — Machine and empty loop

1. Ubuntu 24.04 LTS Desktop if needed; install ROCm, then PyTorch for ROCm.
2. Check GPU:
   - `rocminfo | grep -i gfx`
   - `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`  
     (On AMD, PyTorch still says `cuda`.)
3. Scaffold the repo layout (see [README](../README.md)).
4. Fake `play.py`: noise or still image; keyboard → print action id.
5. EVREN: confirm credits; do not burn them until dynamics week.

## Week 2 — Data and game choice

1. Download **one** game from the DQN Replay Dataset ([offline-rl.github.io](https://offline-rl.github.io/)), not all 60.  
   Candidates only: **DemonAttack** (safer) and **UpNDown** (better demo).  
   Start with ~1–2M steps, not a full 50M-transition dump.  
   If the Google dump is unavailable, use the public Atari parquet fallback (see local `download_atari_data.sh` under hacettepe-studies/AI).
2. Inspect ~50 clips; keep the game where action clearly moves something, background is not chaos, and episodes last more than a few seconds.
3. `dataset.py`: resize **64×64**; contiguous windows of **16** frames; never cross episode boundaries; split **by episode**, 90/10.
4. Small processed set first (~100k–300k frames).
5. Optional extra human/random ALE recordings only after the public set loads.  
   **Lock the game by end of this week.**

## Week 3 — Visual model

1. Train a small conv VAE / regularised AE on the 7800 XT (latent ~**128**, batch 64–128).
2. Every epoch: real \| reconstruction grid.
3. Freeze encoder/decoder before dynamics; do not keep retuning daily.
4. If reconstructions are unreadable: stay at 64×64 or train longer — do not jump architectures.

## Week 4 — Dynamics + first playable

1. Train GRU/LSTM: input `z_t`, action `a_t`, hidden; target `z_{t+1}`; teacher forcing + **8-step** self-rollouts.
2. Pilot locally on a subset; move the full run to EVREN H200 when one-step predictions look right.
3. Wire `play.py`: real start frame → encode → on each key: update latent → decode → show. **No ALE after the start frame.**
4. Record a ~20 s capture (first demo, even if messy).

## Week 5 — Stay alive longer

1. Measure: one-step PSNR / SSIM / LPIPS; error at horizons 1, 8, 32; FPS, latency, VRAM, param count on 7800 XT.
2. Dies under ~2 s: more rollout training, slightly larger GRU, or more diverse actions — **one change at a time**.
3. Blurry but controllable: leave it; playability beats sharpness.
4. Optional extra ALE human play only if left/right/jump are ignored.

## Week 6 — Report and freeze

1. Freeze one checkpoint.
2. Collect: ~1 min play video; real vs predicted rollouts; Week 5 metric table; honest failures (drift, ghost enemies, wall clips).
3. Final report from the proposal structure.
4. Keep the four scripts runnable.

---

## If time is short (early December)

Drop in this order: extra ALE data → horizons > 32 → LPIPS (keep PSNR/SSIM) → Week 5 polish (ship Week 4 playable + honest numbers).

**Never drop:** one chosen game, VAE, dynamics, play loop without the original engine.

## First commands after Ubuntu is up

```bash
rocminfo | grep -E "Marketing Name|GFX"
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Then download ONE game — see https://offline-rl.github.io/
```

## Definition of done

A person presses keys; the model draws the next frames; no game executable is running. It may drift. Measured playability still counts as a finished course project.
