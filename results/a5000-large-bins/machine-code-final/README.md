# CPU inspection of the final large-bin build

Compared `build/custom-only/histogram_bench` with `build/large-bins-overflow/histogram_bench`.

- Old custom device functions: **751**; new: **775**.
- Matching function identities: **751**; resource-record differences: **0**.
- Removed old functions: **0**; added functions: **24**.
- Selected full instruction-word sequences matching: **13/13**.

The 13 selected kernels cover both output clears, scalar and vector shared histograms,
native/u32-local u64 output, the 96 KiB specialization, u8/u32 wide blocks,
partial construction, global accumulation, and scalar/full-tile bitplanes.
The other matching functions have resource comparisons only, including the three partial-reduction specializations.
New narrow-global and shared-prefix-overflow functions are listed separately in [comparison.json](comparison.json).

Exact binary SHA256 hashes, every selected kernel name and encoding hash, normalization
rules, tool version and commands are in that JSON. Complete resource dumps and focused
SASS dumps are retained. Reproduce this CPU-only inspection with `python3 results/a5000-large-bins/machine-code-final/inspect.py`.

This is static preservation evidence, **not proof of equal runtime performance**.
It does not inspect host dispatch or measure execution. No GPU query or launch was performed.
