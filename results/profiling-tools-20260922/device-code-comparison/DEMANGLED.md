# Device instruction comparison with explicit symbol mapping

Independent follow-up to exact mangled-name audit. Match by same cubin filename, architecture and GNU c++filt full demangled function signature; compare unchanged raw instruction byte-size/SHA256 multisets. Preserve and explicitly list all mangled symbol renamings.

| Library | Byte-identical mapped text sections | Renamed signatures | Different instruction sections |
|---|---:|---:|---:|
| count | 782 | 0 | 0 |
| booster | 149 | 48 | 0 |

The initial comparison.json and REPORT.md remain unchanged: the strict mangled-name audit reported 48 removed and 48 added booster identities. All are in higher_order.sm_86.cubin; the compiler-private anonymous-namespace spelling differs. The follow-up retains every original/new name and verifies its same full demangled signature and identical executable section bytes. Other function names match directly.

Counting: 5 cubins, 782 executable sections, 8,822,400 bytes. Booster: 9 cubins, 149 executable sections, 1,721,600 bytes. All mapped compiled device instruction bytes are identical. This claim excludes host code, debug/path metadata, constants, relocation/resource metadata, runtime behavior and performance equivalence. No GPU work was performed.
