# Next bounded choice: E256 model slab

2026-09-23, before implementation. The idle B matrix completed18 cases plus
actual Delicious validation. Every frozen-model comparison passed. Wide forests
benefit, but the three-tree model does not show a reliable complete-call gain.
Existing host code allocates/uploads base, nodes, descriptors and offsets
separately. Select E256 as the next independent candidate: combine those four
model regions into one aligned host/device slab while invoking the identical
ordered-forest kernel. Retain B and per-tree paths as explicit references.

The full numerical/device/memory/lifetime contract, alternatives and fair
experiment are in `training/INFERENCE_MODEL_SLAB_EXPERIMENT.md`, written before
this selection. Charge256-byte per-region alignment and all padding to host
packing, H2D bytes and the GPU budget. Pack directly into the one host slab;
avoid an additional full forest copy. Overflow checks precede allocation and
device pointer arithmetic. No kernel, objective, quantization, tree-add order,
counting path or default changes are part of E. CUDA runtime infrastructure only.

E is a bounded alternative to metadata batching D and an explicit immutable
resident predictor. It is selected first because it can reduce three
malloc/free/upload operations without altering device code or quantizer
synchronization. This is a work-removal hypothesis, not a speed ranking.

Introduce explicit `fused_output_slab`. Compare E directly against B in the same
binary, plus per-tree controls, including the tiny model, large rows and wide
outputs. Keep all existing exact frozen-model tests and add the new policy to
those checks. Recheck padding/init/lifetime and tight-memory behavior; model
slab padding may change the quantizer's feasible tile width. Use at least15
alternating pairs after3 warmups, complete call timing and recorded source/
binary identities, on the idle GPU without heavy CPU work. Existing AB source
and binaries are preserved in `candidate-ab-source/` before any E edit.
