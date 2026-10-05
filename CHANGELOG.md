# 0.1.0 — Public MIT release

The first public release packages the existing C++23/CUDA decision-program system and its Lean proof sources behind one public `decision-programs` command.

- Portable Linux/WSL2 CMake build, install layout and host CI.
- Native XGBoost training, saved-model conversion, exact simplification, prediction, explicit equations and decision traces.
- Study workflows for fixed trials, declared hyperparameter search and out-of-fold nonlinear combination.
- Optional conversion checkpoints, resumable study state, coverage reporting, conditional work estimates and supported proof-search modules.
- The existing qualified categorical RL research integration, with its specialized input domain stated explicitly.
- A generated README poster matching Brian Mingus's pinned-project collection, with accessible text and a linked guide.

This release exposes the current research capabilities. The 448-tree Forest conversion remains incomplete; arbitrary models are not guaranteed to fit within a fixed memory or time budget. RL integration is available without claiming a measured advantage over its control. Dependencies and supported formats are documented in the guide.