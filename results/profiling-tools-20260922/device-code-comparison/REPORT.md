# Compiled device instruction comparison

Extract all CUDA ELF cubins; parse ELF64 section tables; compare byte-size/SHA256 multisets of every executable .text section grouped by exact mangled function name and architecture. No metadata/whitespace normalization of instruction bytes.

SASS machine instruction sections only. Host code/debug info and complete cubin/archive hashes may differ. This does not independently certify constant-data/resource metadata, source semantics or performance equivalence.

| Library | Original / profile cubins | Original / profile text sections | Matched sections | Text bytes, original / profile | All device instruction bytes equal |
|---|---:|---:|---:|---:|---|
| count | 5 / 5 | 782 / 782 | 782 | 8822400 / 8822400 | True |
| booster | 9 / 9 | 149 / 149 | 101 | 1721600 / 1721600 | False |

Exact library/tool/build snapshot hashes, commands, return codes, raw CUDA ELFs, every function hash and any differences are retained in comparison.json and the four extraction directories. No GPU workload or production source edit was performed.
