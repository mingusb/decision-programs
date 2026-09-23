Input: `/home/b/gpu_histogram/results/a5000-profiled/after.sass.txt`; SHA256 `f75ebff0823ecc320ed612c269e65a2846def0e2f69940d04064c6d985936d78`.

Static instruction sites in complete function bodies, including fallback/tail paths. These counts do not measure executed instructions, spills, occupancy, or runtime.

Scalar comparisons require equal architecture, types, dimensions, update method, partial mode, bin capacity, and shared-memory policy limit. A dash means no matching scalar kernel is present.

| ID | Policy | Load | Registers | Spill store/load bytes | Total | Δtotal vs scalar | LDG8 | LDG32 | LDG64 | LDG128 | Shared atomic | CAS64 | Compare | Branch | Vote | Match | POPC | LDL/STL |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | sm_86 bitplane/bitplane u32→u32 local=u32 t256 i8 r1 cap=48KiB bins≤8 | scalar | 23 | 0/0 | 608 | +0 | 0 | 8 | 0 | 0 | 0 | 0 | 114 | 61 | 33 | 0 | 8 | 0 |
| 2 | sm_86 bitplane/bitplane u32→u32 local=u32 t256 i8 r1 cap=48KiB bins≤8 | full_tile | 26 | 0/0 | 856 | +248 | 0 | 16 | 0 | 0 | 0 | 0 | 156 | 62 | 57 | 0 | 16 | 0 |
| 3 | sm_86 bitplane/bitplane u32→u32 local=u32 t256 i8 r1 cap=48KiB bins≤256 | scalar | 34 | 0/0 | 1320 | +0 | 0 | 8 | 0 | 0 | 0 | 0 | 193 | 61 | 73 | 0 | 64 | 0 |
| 4 | sm_86 bitplane/bitplane u32→u32 local=u32 t256 i8 r1 cap=48KiB bins≤256 | full_tile | 40 | 0/0 | 2016 | +696 | 0 | 16 | 0 | 0 | 0 | 0 | 275 | 62 | 137 | 0 | 128 | 0 |
| 5 | sm_86 bitplane/bitplane u32→u32 local=u32 t256 i16 r1 cap=48KiB bins≤8 | scalar | 23 | 0/0 | 1136 | +0 | 0 | 16 | 0 | 0 | 0 | 0 | 218 | 117 | 65 | 0 | 16 | 0 |
| 6 | sm_86 bitplane/bitplane u32→u32 local=u32 t256 i16 r1 cap=48KiB bins≤8 | full_tile | 34 | 0/0 | 1632 | +496 | 0 | 32 | 0 | 0 | 0 | 0 | 300 | 118 | 113 | 0 | 32 | 0 |
| 7 | sm_86 bitplane/bitplane u32→u32 local=u32 t256 i16 r1 cap=48KiB bins≤256 | scalar | 34 | 0/0 | 2528 | +0 | 0 | 16 | 0 | 0 | 0 | 0 | 377 | 103 | 145 | 0 | 128 | 0 |
| 8 | sm_86 bitplane/bitplane u32→u32 local=u32 t256 i16 r1 cap=48KiB bins≤256 | full_tile | 40 | 0/0 | 3904 | +1376 | 0 | 32 | 0 | 0 | 0 | 0 | 539 | 118 | 273 | 0 | 256 | 0 |
| 9 | sm_86 bitplane/bitplane u32→u64 local=u64 t256 i8 r1 cap=48KiB bins≤8 | scalar | 26 | 0/0 | 624 | +0 | 0 | 8 | 0 | 0 | 0 | 0 | 115 | 61 | 33 | 0 | 8 | 0 |
| 10 | sm_86 bitplane/bitplane u32→u64 local=u64 t256 i8 r1 cap=48KiB bins≤8 | full_tile | 24 | 0/0 | 872 | +248 | 0 | 16 | 0 | 0 | 0 | 0 | 157 | 62 | 57 | 0 | 16 | 0 |
| 11 | sm_86 bitplane/bitplane u32→u64 local=u64 t256 i8 r1 cap=48KiB bins≤256 | scalar | 44 | 0/0 | 1408 | +0 | 0 | 8 | 0 | 0 | 0 | 0 | 194 | 61 | 73 | 0 | 64 | 0 |
| 12 | sm_86 bitplane/bitplane u32→u64 local=u64 t256 i8 r1 cap=48KiB bins≤256 | full_tile | 48 | 0/0 | 2120 | +712 | 0 | 16 | 0 | 0 | 0 | 0 | 276 | 62 | 137 | 0 | 128 | 0 |
| 13 | sm_86 bitplane/bitplane u32→u64 local=u64 t256 i16 r1 cap=48KiB bins≤8 | scalar | 23 | 0/0 | 1160 | +0 | 0 | 16 | 0 | 0 | 0 | 0 | 219 | 117 | 65 | 0 | 16 | 0 |
| 14 | sm_86 bitplane/bitplane u32→u64 local=u64 t256 i16 r1 cap=48KiB bins≤8 | full_tile | 32 | 0/0 | 1664 | +504 | 0 | 32 | 0 | 0 | 0 | 0 | 301 | 118 | 113 | 0 | 32 | 0 |
| 15 | sm_86 bitplane/bitplane u32→u64 local=u64 t256 i16 r1 cap=48KiB bins≤256 | scalar | 44 | 0/0 | 2680 | +0 | 0 | 16 | 0 | 0 | 0 | 0 | 378 | 103 | 145 | 0 | 128 | 0 |
| 16 | sm_86 bitplane/bitplane u32→u64 local=u64 t256 i16 r1 cap=48KiB bins≤256 | full_tile | 48 | 0/0 | 4112 | +1432 | 0 | 32 | 0 | 0 | 0 | 0 | 540 | 118 | 273 | 0 | 256 | 0 |
| 17 | sm_86 bitplane/bitplane u8→u32 local=u32 t256 i8 r1 cap=48KiB bins≤8 | scalar | 23 | 0/0 | 608 | +0 | 8 | 0 | 0 | 0 | 0 | 0 | 114 | 61 | 33 | 0 | 8 | 0 |
| 18 | sm_86 bitplane/bitplane u8→u32 local=u32 t256 i8 r1 cap=48KiB bins≤8 | full_tile | 26 | 0/0 | 856 | +248 | 16 | 0 | 0 | 0 | 0 | 0 | 156 | 62 | 57 | 0 | 16 | 0 |
| 19 | sm_86 bitplane/bitplane u8→u32 local=u32 t256 i8 r1 cap=48KiB bins≤256 | scalar | 34 | 0/0 | 1320 | +0 | 8 | 0 | 0 | 0 | 0 | 0 | 193 | 61 | 73 | 0 | 64 | 0 |
| 20 | sm_86 bitplane/bitplane u8→u32 local=u32 t256 i8 r1 cap=48KiB bins≤256 | full_tile | 39 | 0/0 | 2008 | +688 | 16 | 0 | 0 | 0 | 0 | 0 | 275 | 62 | 137 | 0 | 128 | 0 |
| 21 | sm_86 bitplane/bitplane u8→u32 local=u32 t256 i16 r1 cap=48KiB bins≤8 | scalar | 23 | 0/0 | 1136 | +0 | 16 | 0 | 0 | 0 | 0 | 0 | 218 | 117 | 65 | 0 | 16 | 0 |
| 22 | sm_86 bitplane/bitplane u8→u32 local=u32 t256 i16 r1 cap=48KiB bins≤8 | full_tile | 34 | 0/0 | 1624 | +488 | 32 | 0 | 0 | 0 | 0 | 0 | 300 | 118 | 113 | 0 | 32 | 0 |
| 23 | sm_86 bitplane/bitplane u8→u32 local=u32 t256 i16 r1 cap=48KiB bins≤256 | scalar | 34 | 0/0 | 2528 | +0 | 16 | 0 | 0 | 0 | 0 | 0 | 377 | 103 | 145 | 0 | 128 | 0 |
| 24 | sm_86 bitplane/bitplane u8→u32 local=u32 t256 i16 r1 cap=48KiB bins≤256 | full_tile | 39 | 0/0 | 3896 | +1368 | 32 | 0 | 0 | 0 | 0 | 0 | 539 | 118 | 273 | 0 | 256 | 0 |
| 25 | sm_86 bitplane/bitplane u8→u64 local=u64 t256 i8 r1 cap=48KiB bins≤8 | scalar | 26 | 0/0 | 624 | +0 | 8 | 0 | 0 | 0 | 0 | 0 | 115 | 61 | 33 | 0 | 8 | 0 |
| 26 | sm_86 bitplane/bitplane u8→u64 local=u64 t256 i8 r1 cap=48KiB bins≤8 | full_tile | 24 | 0/0 | 872 | +248 | 16 | 0 | 0 | 0 | 0 | 0 | 157 | 62 | 57 | 0 | 16 | 0 |
| 27 | sm_86 bitplane/bitplane u8→u64 local=u64 t256 i8 r1 cap=48KiB bins≤256 | scalar | 44 | 0/0 | 1400 | +0 | 8 | 0 | 0 | 0 | 0 | 0 | 194 | 61 | 73 | 0 | 64 | 0 |
| 28 | sm_86 bitplane/bitplane u8→u64 local=u64 t256 i8 r1 cap=48KiB bins≤256 | full_tile | 48 | 0/0 | 2128 | +728 | 16 | 0 | 0 | 0 | 0 | 0 | 276 | 62 | 137 | 0 | 128 | 0 |
| 29 | sm_86 bitplane/bitplane u8→u64 local=u64 t256 i16 r1 cap=48KiB bins≤8 | scalar | 23 | 0/0 | 1160 | +0 | 16 | 0 | 0 | 0 | 0 | 0 | 219 | 117 | 65 | 0 | 16 | 0 |
| 30 | sm_86 bitplane/bitplane u8→u64 local=u64 t256 i16 r1 cap=48KiB bins≤8 | full_tile | 32 | 0/0 | 1656 | +496 | 32 | 0 | 0 | 0 | 0 | 0 | 301 | 118 | 113 | 0 | 32 | 0 |
| 31 | sm_86 bitplane/bitplane u8→u64 local=u64 t256 i16 r1 cap=48KiB bins≤256 | scalar | 44 | 0/0 | 2680 | +0 | 16 | 0 | 0 | 0 | 0 | 0 | 378 | 103 | 145 | 0 | 128 | 0 |
| 32 | sm_86 bitplane/bitplane u8→u64 local=u64 t256 i16 r1 cap=48KiB bins≤256 | full_tile | 48 | 0/0 | 4112 | +1432 | 32 | 0 | 0 | 0 | 0 | 0 | 540 | 118 | 273 | 0 | 256 | 0 |

PTXAS log: `/home/b/gpu_histogram/results/a5000-profiled/build.log`; SHA256 `d004bbf150bb4eeedde5dd8303fc000e388db7607ea67d7496457479260bf5bb`. Parsed 772 function records; 0 report nonzero spill stores or loads.

Exact function identities:

1. `void gh::detail::(anonymous namespace)::bitplane_histogram<unsigned int, unsigned int, 8u, 256, 8>(unsigned int const*, unsigned long, unsigned int*, unsigned int)`
2. `void gh::detail::(anonymous namespace)::bitplane_histogram_full_tile<unsigned int, unsigned int, 8u, 256, 8>(unsigned int const*, unsigned long, unsigned int*, unsigned int)`
3. `void gh::detail::(anonymous namespace)::bitplane_histogram<unsigned int, unsigned int, 256u, 256, 8>(unsigned int const*, unsigned long, unsigned int*, unsigned int)`
4. `void gh::detail::(anonymous namespace)::bitplane_histogram_full_tile<unsigned int, unsigned int, 256u, 256, 8>(unsigned int const*, unsigned long, unsigned int*, unsigned int)`
5. `void gh::detail::(anonymous namespace)::bitplane_histogram<unsigned int, unsigned int, 8u, 256, 16>(unsigned int const*, unsigned long, unsigned int*, unsigned int)`
6. `void gh::detail::(anonymous namespace)::bitplane_histogram_full_tile<unsigned int, unsigned int, 8u, 256, 16>(unsigned int const*, unsigned long, unsigned int*, unsigned int)`
7. `void gh::detail::(anonymous namespace)::bitplane_histogram<unsigned int, unsigned int, 256u, 256, 16>(unsigned int const*, unsigned long, unsigned int*, unsigned int)`
8. `void gh::detail::(anonymous namespace)::bitplane_histogram_full_tile<unsigned int, unsigned int, 256u, 256, 16>(unsigned int const*, unsigned long, unsigned int*, unsigned int)`
9. `void gh::detail::(anonymous namespace)::bitplane_histogram<unsigned int, unsigned long long, 8u, 256, 8>(unsigned int const*, unsigned long, unsigned long long*, unsigned int)`
10. `void gh::detail::(anonymous namespace)::bitplane_histogram_full_tile<unsigned int, unsigned long long, 8u, 256, 8>(unsigned int const*, unsigned long, unsigned long long*, unsigned int)`
11. `void gh::detail::(anonymous namespace)::bitplane_histogram<unsigned int, unsigned long long, 256u, 256, 8>(unsigned int const*, unsigned long, unsigned long long*, unsigned int)`
12. `void gh::detail::(anonymous namespace)::bitplane_histogram_full_tile<unsigned int, unsigned long long, 256u, 256, 8>(unsigned int const*, unsigned long, unsigned long long*, unsigned int)`
13. `void gh::detail::(anonymous namespace)::bitplane_histogram<unsigned int, unsigned long long, 8u, 256, 16>(unsigned int const*, unsigned long, unsigned long long*, unsigned int)`
14. `void gh::detail::(anonymous namespace)::bitplane_histogram_full_tile<unsigned int, unsigned long long, 8u, 256, 16>(unsigned int const*, unsigned long, unsigned long long*, unsigned int)`
15. `void gh::detail::(anonymous namespace)::bitplane_histogram<unsigned int, unsigned long long, 256u, 256, 16>(unsigned int const*, unsigned long, unsigned long long*, unsigned int)`
16. `void gh::detail::(anonymous namespace)::bitplane_histogram_full_tile<unsigned int, unsigned long long, 256u, 256, 16>(unsigned int const*, unsigned long, unsigned long long*, unsigned int)`
17. `void gh::detail::(anonymous namespace)::bitplane_histogram<unsigned char, unsigned int, 8u, 256, 8>(unsigned char const*, unsigned long, unsigned int*, unsigned int)`
18. `void gh::detail::(anonymous namespace)::bitplane_histogram_full_tile<unsigned char, unsigned int, 8u, 256, 8>(unsigned char const*, unsigned long, unsigned int*, unsigned int)`
19. `void gh::detail::(anonymous namespace)::bitplane_histogram<unsigned char, unsigned int, 256u, 256, 8>(unsigned char const*, unsigned long, unsigned int*, unsigned int)`
20. `void gh::detail::(anonymous namespace)::bitplane_histogram_full_tile<unsigned char, unsigned int, 256u, 256, 8>(unsigned char const*, unsigned long, unsigned int*, unsigned int)`
21. `void gh::detail::(anonymous namespace)::bitplane_histogram<unsigned char, unsigned int, 8u, 256, 16>(unsigned char const*, unsigned long, unsigned int*, unsigned int)`
22. `void gh::detail::(anonymous namespace)::bitplane_histogram_full_tile<unsigned char, unsigned int, 8u, 256, 16>(unsigned char const*, unsigned long, unsigned int*, unsigned int)`
23. `void gh::detail::(anonymous namespace)::bitplane_histogram<unsigned char, unsigned int, 256u, 256, 16>(unsigned char const*, unsigned long, unsigned int*, unsigned int)`
24. `void gh::detail::(anonymous namespace)::bitplane_histogram_full_tile<unsigned char, unsigned int, 256u, 256, 16>(unsigned char const*, unsigned long, unsigned int*, unsigned int)`
25. `void gh::detail::(anonymous namespace)::bitplane_histogram<unsigned char, unsigned long long, 8u, 256, 8>(unsigned char const*, unsigned long, unsigned long long*, unsigned int)`
26. `void gh::detail::(anonymous namespace)::bitplane_histogram_full_tile<unsigned char, unsigned long long, 8u, 256, 8>(unsigned char const*, unsigned long, unsigned long long*, unsigned int)`
27. `void gh::detail::(anonymous namespace)::bitplane_histogram<unsigned char, unsigned long long, 256u, 256, 8>(unsigned char const*, unsigned long, unsigned long long*, unsigned int)`
28. `void gh::detail::(anonymous namespace)::bitplane_histogram_full_tile<unsigned char, unsigned long long, 256u, 256, 8>(unsigned char const*, unsigned long, unsigned long long*, unsigned int)`
29. `void gh::detail::(anonymous namespace)::bitplane_histogram<unsigned char, unsigned long long, 8u, 256, 16>(unsigned char const*, unsigned long, unsigned long long*, unsigned int)`
30. `void gh::detail::(anonymous namespace)::bitplane_histogram_full_tile<unsigned char, unsigned long long, 8u, 256, 16>(unsigned char const*, unsigned long, unsigned long long*, unsigned int)`
31. `void gh::detail::(anonymous namespace)::bitplane_histogram<unsigned char, unsigned long long, 256u, 256, 16>(unsigned char const*, unsigned long, unsigned long long*, unsigned int)`
32. `void gh::detail::(anonymous namespace)::bitplane_histogram_full_tile<unsigned char, unsigned long long, 256u, 256, 16>(unsigned char const*, unsigned long, unsigned long long*, unsigned int)`
