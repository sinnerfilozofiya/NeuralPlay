# NeuralPlay

**Learning a playable game engine from pixels and actions** — AIE623 (Artificial Intelligence) course project.

At play time the original engine is gone. A compact neural model reads player actions and draws the next frames for **one** 2D side-scrolling Atari level.

## Goal

Build a complete prototype: data → visual model → action-conditioned dynamics → real-time play loop → measured evaluation.

Success means a person can sit at the machine, press keys, and the learned model draws the next frames with **no game executable running**. Drift after a few seconds is acceptable if it is measured honestly.

This is **not** a multi-game model, a diffusion engine, or a commercial replacement. Scope stays at one chosen game / level.

## Approach (short)

1. **Data** — trajectories from the [DQN Replay Dataset](https://offline-rl.github.io/) (or a public Atari fallback), one game only (candidates: DemonAttack, UpNDown).
2. **Visual model** — convolutional VAE / regularised autoencoder → latent frames (~64×64).
3. **Dynamics** — GRU/LSTM: `(z_t, a_t, h_t) → z_{t+1}`, teacher forcing + short self-rollouts.
4. **Play** — encode one real start frame; then only model updates; keyboard uses the same discrete action set as training.
5. **Eval** — one-step and multi-horizon image metrics, control checks, playability seconds, FPS / latency / VRAM on local AMD RX 7800 XT (heavy runs on EVREN H200 when needed).

## Docs

| Doc | What it is |
| --- | --- |
| [docs/PROPOSAL.md](docs/PROPOSAL.md) | Problem, method, evaluation, contribution (from the AIE623 proposal) |
| [docs/PLAN.md](docs/PLAN.md) | Week-by-week starting plan through the December deadline |

## Intended layout (to implement)

```text
neuralplay/
  data/raw/           # downloaded replay shards
  data/processed/     # train windows
  src/models.py       # encoder, decoder, recurrent dynamics
  src/dataset.py
  scripts/train_vae.py
  scripts/train_dynamics.py
  scripts/play.py
  scripts/eval.py
  requirements.txt
```

## Definition of done

Demo video + metric table + short report. Freeze one checkpoint. Keep `train_vae.py`, `train_dynamics.py`, `play.py`, and `eval.py` runnable.

## Sources used for this documentation

Local drafts under `~/Desktop/hacettepe-studies/AI/`:

- `AIE623_proposal_neuralplay.md` — course proposal
- `NEURALPLAY_PLAN.md` — week plan (due December 2026)
- supporting scripts: `download_atari_data.sh`, `make_neuralplay_proposal.py`
