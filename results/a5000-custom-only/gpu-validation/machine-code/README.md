# CPU inspection of preserved kernels

All **751 custom device functions** are present in both binaries, with identical reported resource records. The complete instruction-word sequences of **13 selected device kernels** are identical. The selection includes both output clears; scalar and vector shared kernels; native/u32-local 64-bit output; the 96 KiB opt-in specialization; byte and u32 wide-block paths; partial construction; global accumulation; and scalar/full-tile bitplanes.

This compares `build/defaults/histogram_bench` with `build/histogram_bench`. Exact SHA256 hashes, all 13 names, per-kernel encoding hashes, comparison rules and reproducible commands are in [comparison.json](comparison.json). Full resource records and the focused SASS dumps are preserved alongside it. The other 738 kernels have matching resource records but were not compared instruction by instruction.

Both old and new host dispatch functions already call `gh::supported` out of line. Moving validation into `gh_policy` therefore introduced no new validation-call boundary. The old call is at 0x57629 and the new call at 0x5479d; [old](old-dispatch.asm) and [new](new-dispatch.asm) host disassembly preserve the evidence. Host branches and code layout changed when reference paths were removed.

Kernel identity is matched by basename and complete template arguments, ignoring only the translation-unit anonymous-namespace prefix. Encoding comparison includes every emitted instruction/control word in order, excluding textual module names and addresses. No GPU execution or device query was used for this inspection.

Static preservation does not establish equal runtime performance. Paired measurements must specify identical algorithms, tuning, grid, local counters, clearing policy and benchmark modes. Automatic-selection behavior is a separate change. Enum ordinals changed when benchmark-only algorithms left the public API; callers must rebuild against the new header and library.
