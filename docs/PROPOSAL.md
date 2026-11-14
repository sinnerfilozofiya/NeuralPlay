# NeuralPlay: Learning a Playable Game Engine from Pixels and Actions

**Course:** AIE623 – Artificial Intelligence  
**Date:** 25 October 2026  

Source: local draft `~/Desktop/hacettepe-studies/AI/AIE623_proposal_neuralplay.md`.

---

## 1. Problem definition

A conventional 2D game engine is a hand-written loop: it updates an internal state from player input, then renders that state as pixels. This project asks whether a compact neural model, trained on recorded frames and the actions that produced them, can stand in for that loop on one complete level.

Predicting a single next frame is not enough. One-step prediction can look convincing while the model is still shown real frames. A playable engine has to continue after the original program is removed: the player supplies actions, the model updates its own latent state, a decoder turns that state into the next image, and the prediction becomes the next input. Small mistakes then compound — characters drift, collisions fail, and the level stops behaving like the game it was trained on.

**Scope:** one 2D side-scrolling environment and one complete level or stage. Not a general-purpose generator, multi-game model, or commercial replacement. After training, a person should be able to play the learned level without calling the original engine.

## 2. Motivation

Game engines are usually written as rules for rendering, physics, collisions, enemies, and camera motion. A learned engine treats the same problem as a data problem: recover visual dynamics and the effect of actions from gameplay. That is a concrete test of whether a world model has learned the game, not only how to reconstruct a picture. A model that copies pixels but ignores a jump, a hit, or a scroll is not an engine.

If a small model can stay interactive on a consumer GPU, the result is a complete system — collect data, train, then play — rather than an offline video demo. This is a course implementation study. The value is to build the loop and measure how long the learned level remains playable before it drifts.

## 3. Dataset

Main dataset: **[DQN Replay Dataset](https://offline-rl.github.io/)** (Agarwal, Schuurmans & Norouzi, ICML 2020) — logs of DQN agents on 60 Atari 2600 games. For each game, five runs; each run covers ~200M environment frames (~50M transitions with action repeat of four).

Use **one** 2D side-scrolling game after a short check of visual complexity, action influence, and episode length. Extract contiguous frame–action sequences for a single level or stage. The original engine is a data source and later an evaluation reference only. If public logs miss common player behaviours, collect extra Arcade Learning Environment trajectories (scripted, random, human).

Frames resize to about **64×64** so training and real-time play fit on a 16 GB consumer GPU. A short reserved H200 session can cover heavier training if needed.

## 4. Proposed methodology

Compact world-model pattern of Ha and Schmidhuber (2018): visual encoder, action-conditioned recurrent dynamics, decoder.

- Trajectories → fixed-length windows that never cross episode boundaries.
- Each sample: short sequence of low-res RGB frames + discrete actions.
- Train/test split **by episode**.

**Visual model:** convolutional encoder–decoder (VAE or regularised AE) compresses each frame into a small latent and reconstructs it.

**Dynamics:** GRU or LSTM takes current latent, player action, and hidden state → next latent. Teacher forcing on real latents, plus short self-rollouts to reduce drift (training detail, not a second method to compare).

**Inference:** original engine gone. Encode one real starting frame, then: read action → update latent → decode → display → feed latent back. Keyboard/gamepad uses the same discrete action set as training. PyTorch; target real-time play on AMD RX 7800 XT. No external generative APIs.

## 5. Evaluation

Evaluate as a playable engine, not only as an image predictor:

- Held-out one-step: PSNR, SSIM, LPIPS.
- Stability: start from a real frame, replay recorded actions; error at horizons 1, 8, 32, longer.
- Control: movement/jump produce expected motion; player, obstacles, enemies stay consistent (no vanish / duplicate / wall-clip).
- Playability: seconds or screens the level stays controllable; whether a short human session can make visible progress.
- Efficiency on RX 7800 XT: FPS, input-to-display latency, peak VRAM, model size.

A useful result can still be a model that plays well for a limited horizon and then drifts.

## 6. Previous methods

Ha and Schmidhuber (2018) trained a VAE and action-conditioned recurrent model for agents acting in a “dream.” Dreamer (Hafner et al., 2020, 2023) uses latent dynamics mainly to train policies. Recent neural engines such as GameNGen (Valevski et al., 2025) and DIAMOND (Alonso et al., 2024) show gameplay-trained models can replace a programmed loop at interactive rates; those systems are heavier than this course project and are motivation only.

**No new architecture claimed.** Apply the lighter encoder–recurrent–decoder design to one side-scrolling level, with the requirement that a person can play after the original engine is removed.

## 7. Contribution and limitations

**Contribution:** complete prototype — dataset subset, trained visual and dynamics models, real-time play loop, short evaluation of visual quality, control, stability, and speed.

**Not claimed:** new method, multi-game engine, or long-horizon fidelity matching the original program.

**Main limitation:** compounding error. The project succeeds if the learned level is recognisable, controllable in real time, and measured clearly, even if it cannot be played indefinitely.

---

## References

- Agarwal, R., Schuurmans, D., & Norouzi, M. (2020). [An optimistic perspective on offline reinforcement learning](https://proceedings.mlr.press/v119/agarwal20c.html). *ICML*.
- Alonso, E., et al. (2024). [Diffusion for world modeling: Visual details matter in Atari](https://arxiv.org/abs/2405.12399). *NeurIPS*.
- Ha, D., & Schmidhuber, J. (2018). [Recurrent world models facilitate policy evolution](https://worldmodels.github.io/). *NeurIPS*.
- Hafner, D., et al. (2020). [Dream to control: Learning behaviors by latent imagination](https://arxiv.org/abs/1912.01603). *ICLR*.
- Hafner, D., et al. (2023). [Mastering diverse domains through world models](https://arxiv.org/abs/2301.04104). *arXiv:2301.04104*.
- Valevski, D., et al. (2025). [Diffusion models are real-time game engines](https://arxiv.org/abs/2408.14837). *ICLR*.
