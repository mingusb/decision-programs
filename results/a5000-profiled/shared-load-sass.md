Input: `/home/b/gpu_histogram/results/a5000-profiled/after.sass.txt`; SHA256 `f75ebff0823ecc320ed612c269e65a2846def0e2f69940d04064c6d985936d78`.

Static instruction sites in complete function bodies, including fallback/tail paths. These counts do not measure executed instructions, spills, occupancy, or runtime.

Scalar comparisons require equal architecture, types, dimensions, update method, partial mode, bin capacity, and shared-memory policy limit. A dash means no matching scalar kernel is present.

| ID | Policy | Load | Registers | Spill store/load bytes | Total | Δtotal vs scalar | LDG8 | LDG32 | LDG64 | LDG128 | Shared atomic | CAS64 | Compare | Branch | Vote | Match | POPC | LDL/STL |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | sm_86 shared/atomic u32→u32 local=u32 t256 i8 r1 cap=48KiB | scalar | 32 | 0/0 | 168 | +0 | 0 | 8 | 0 | 0 | 8 | 0 | 34 | 16 | 0 | 0 | 0 | 0 |
| 2 | sm_86 shared/atomic u32→u32 local=u32 t256 i8 r1 cap=48KiB | full_tile | 32 | 0/0 | 192 | +24 | 0 | 16 | 0 | 0 | 16 | 0 | 36 | 18 | 0 | 0 | 0 | 0 |
| 3 | sm_86 shared/atomic u32→u32 local=u32 t256 i8 r1 cap=48KiB | vector4 | 32 | 0/0 | 304 | +136 | 0 | 24 | 0 | 2 | 32 | 0 | 59 | 31 | 0 | 0 | 0 | 0 |
| 4 | sm_86 shared/atomic u32→u32 local=u32 t256 i8 r1 cap=96KiB | vector4 | 32 | 0/0 | 304 | — | 0 | 24 | 0 | 2 | 32 | 0 | 59 | 31 | 0 | 0 | 0 | 0 |
| 5 | sm_86 shared/atomic u32→u32 local=u32 t256 i16 r1 cap=48KiB | scalar | 52 | 0/0 | 296 | +0 | 0 | 16 | 0 | 0 | 16 | 0 | 74 | 24 | 0 | 0 | 0 | 0 |
| 6 | sm_86 shared/atomic u32→u32 local=u32 t256 i16 r1 cap=48KiB | full_tile | 51 | 0/0 | 328 | +32 | 0 | 32 | 0 | 0 | 32 | 0 | 77 | 26 | 0 | 0 | 0 | 0 |
| 7 | sm_86 shared/atomic u32→u32 local=u32 t256 i16 r1 cap=48KiB | vector4 | 56 | 0/0 | 592 | +296 | 0 | 48 | 0 | 4 | 64 | 0 | 143 | 47 | 0 | 0 | 0 | 0 |
| 8 | sm_86 shared/atomic u32→u64 local=u32 t256 i8 r1 cap=48KiB | scalar | 32 | 0/0 | 168 | +0 | 0 | 8 | 0 | 0 | 8 | 0 | 34 | 16 | 0 | 0 | 0 | 0 |
| 9 | sm_86 shared/atomic u32→u64 local=u32 t256 i8 r1 cap=48KiB | full_tile | 32 | 0/0 | 192 | +24 | 0 | 16 | 0 | 0 | 16 | 0 | 36 | 18 | 0 | 0 | 0 | 0 |
| 10 | sm_86 shared/atomic u32→u64 local=u32 t256 i8 r1 cap=48KiB | vector4 | 32 | 0/0 | 304 | +136 | 0 | 24 | 0 | 2 | 32 | 0 | 59 | 31 | 0 | 0 | 0 | 0 |
| 11 | sm_86 shared/atomic u32→u64 local=u32 t256 i8 r1 cap=96KiB | vector4 | 32 | 0/0 | 304 | — | 0 | 24 | 0 | 2 | 32 | 0 | 59 | 31 | 0 | 0 | 0 | 0 |
| 12 | sm_86 shared/atomic u32→u64 local=u32 t256 i16 r1 cap=48KiB | scalar | 52 | 0/0 | 296 | +0 | 0 | 16 | 0 | 0 | 16 | 0 | 74 | 24 | 0 | 0 | 0 | 0 |
| 13 | sm_86 shared/atomic u32→u64 local=u32 t256 i16 r1 cap=48KiB | full_tile | 51 | 0/0 | 328 | +32 | 0 | 32 | 0 | 0 | 32 | 0 | 77 | 26 | 0 | 0 | 0 | 0 |
| 14 | sm_86 shared/atomic u32→u64 local=u32 t256 i16 r1 cap=48KiB | vector4 | 56 | 0/0 | 592 | +296 | 0 | 48 | 0 | 4 | 64 | 0 | 143 | 47 | 0 | 0 | 0 | 0 |
| 15 | sm_86 shared/atomic u32→u64 local=u64 t256 i8 r1 cap=48KiB | scalar | 32 | 0/0 | 224 | +0 | 0 | 8 | 0 | 0 | 8 | 8 | 51 | 24 | 0 | 0 | 0 | 0 |
| 16 | sm_86 shared/atomic u32→u64 local=u64 t256 i8 r1 cap=48KiB | full_tile | 32 | 0/0 | 320 | +96 | 0 | 16 | 0 | 0 | 16 | 16 | 69 | 34 | 0 | 0 | 0 | 0 |
| 17 | sm_86 shared/atomic u32→u64 local=u64 t256 i8 r1 cap=48KiB | vector4 | 34 | 0/0 | 568 | +344 | 0 | 24 | 0 | 2 | 32 | 32 | 124 | 63 | 0 | 0 | 0 | 0 |
| 18 | sm_86 shared/atomic u32→u64 local=u64 t256 i8 r1 cap=96KiB | vector4 | 34 | 0/0 | 568 | — | 0 | 24 | 0 | 2 | 32 | 32 | 124 | 63 | 0 | 0 | 0 | 0 |
| 19 | sm_86 shared/atomic u32→u64 local=u64 t256 i16 r1 cap=48KiB | scalar | 52 | 0/0 | 408 | +0 | 0 | 16 | 0 | 0 | 16 | 16 | 107 | 40 | 0 | 0 | 0 | 0 |
| 20 | sm_86 shared/atomic u32→u64 local=u64 t256 i16 r1 cap=48KiB | full_tile | 51 | 0/0 | 584 | +176 | 0 | 32 | 0 | 0 | 32 | 32 | 142 | 58 | 0 | 0 | 0 | 0 |
| 21 | sm_86 shared/atomic u32→u64 local=u64 t256 i16 r1 cap=48KiB | vector4 | 57 | 0/0 | 1104 | +696 | 0 | 48 | 0 | 4 | 64 | 64 | 264 | 111 | 0 | 0 | 0 | 0 |
| 22 | sm_86 shared/atomic u8→u32 local=u32 t256 i8 r1 cap=48KiB | scalar | 32 | 0/0 | 168 | +0 | 8 | 0 | 0 | 0 | 8 | 0 | 34 | 16 | 0 | 0 | 0 | 0 |
| 23 | sm_86 shared/atomic u8→u32 local=u32 t256 i8 r1 cap=48KiB | full_tile | 32 | 0/0 | 184 | +16 | 16 | 0 | 0 | 0 | 16 | 0 | 36 | 18 | 0 | 0 | 0 | 0 |
| 24 | sm_86 shared/atomic u8→u32 local=u32 t256 i8 r1 cap=48KiB | vector4 | 32 | 0/0 | 312 | +144 | 24 | 2 | 0 | 0 | 32 | 0 | 59 | 31 | 0 | 0 | 0 | 0 |
| 25 | sm_86 shared/atomic u8→u32 local=u32 t256 i8 r1 cap=96KiB | vector4 | 32 | 0/0 | 312 | — | 24 | 2 | 0 | 0 | 32 | 0 | 59 | 31 | 0 | 0 | 0 | 0 |
| 26 | sm_86 shared/atomic u8→u32 local=u32 t256 i16 r1 cap=48KiB | scalar | 52 | 0/0 | 296 | +0 | 16 | 0 | 0 | 0 | 16 | 0 | 74 | 24 | 0 | 0 | 0 | 0 |
| 27 | sm_86 shared/atomic u8→u32 local=u32 t256 i16 r1 cap=48KiB | full_tile | 51 | 0/0 | 328 | +32 | 32 | 0 | 0 | 0 | 32 | 0 | 77 | 26 | 0 | 0 | 0 | 0 |
| 28 | sm_86 shared/atomic u8→u32 local=u32 t256 i16 r1 cap=48KiB | vector4 | 56 | 0/0 | 624 | +328 | 48 | 4 | 0 | 0 | 64 | 0 | 143 | 47 | 0 | 0 | 0 | 0 |
| 29 | sm_86 shared/atomic u8→u64 local=u32 t256 i8 r1 cap=48KiB | scalar | 32 | 0/0 | 168 | +0 | 8 | 0 | 0 | 0 | 8 | 0 | 34 | 16 | 0 | 0 | 0 | 0 |
| 30 | sm_86 shared/atomic u8→u64 local=u32 t256 i8 r1 cap=48KiB | full_tile | 32 | 0/0 | 184 | +16 | 16 | 0 | 0 | 0 | 16 | 0 | 36 | 18 | 0 | 0 | 0 | 0 |
| 31 | sm_86 shared/atomic u8→u64 local=u32 t256 i8 r1 cap=48KiB | vector4 | 32 | 0/0 | 320 | +152 | 24 | 2 | 0 | 0 | 32 | 0 | 59 | 31 | 0 | 0 | 0 | 0 |
| 32 | sm_86 shared/atomic u8→u64 local=u32 t256 i8 r1 cap=96KiB | vector4 | 32 | 0/0 | 320 | — | 24 | 2 | 0 | 0 | 32 | 0 | 59 | 31 | 0 | 0 | 0 | 0 |
| 33 | sm_86 shared/atomic u8→u64 local=u32 t256 i16 r1 cap=48KiB | scalar | 52 | 0/0 | 296 | +0 | 16 | 0 | 0 | 0 | 16 | 0 | 74 | 24 | 0 | 0 | 0 | 0 |
| 34 | sm_86 shared/atomic u8→u64 local=u32 t256 i16 r1 cap=48KiB | full_tile | 51 | 0/0 | 328 | +32 | 32 | 0 | 0 | 0 | 32 | 0 | 77 | 26 | 0 | 0 | 0 | 0 |
| 35 | sm_86 shared/atomic u8→u64 local=u32 t256 i16 r1 cap=48KiB | vector4 | 56 | 0/0 | 624 | +328 | 48 | 4 | 0 | 0 | 64 | 0 | 143 | 47 | 0 | 0 | 0 | 0 |
| 36 | sm_86 shared/atomic u8→u64 local=u64 t256 i8 r1 cap=48KiB | scalar | 32 | 0/0 | 224 | +0 | 8 | 0 | 0 | 0 | 8 | 8 | 51 | 24 | 0 | 0 | 0 | 0 |
| 37 | sm_86 shared/atomic u8→u64 local=u64 t256 i8 r1 cap=48KiB | full_tile | 32 | 0/0 | 320 | +96 | 16 | 0 | 0 | 0 | 16 | 16 | 69 | 34 | 0 | 0 | 0 | 0 |
| 38 | sm_86 shared/atomic u8→u64 local=u64 t256 i8 r1 cap=48KiB | vector4 | 34 | 0/0 | 568 | +344 | 24 | 2 | 0 | 0 | 32 | 32 | 124 | 63 | 0 | 0 | 0 | 0 |
| 39 | sm_86 shared/atomic u8→u64 local=u64 t256 i8 r1 cap=96KiB | vector4 | 34 | 0/0 | 568 | — | 24 | 2 | 0 | 0 | 32 | 32 | 124 | 63 | 0 | 0 | 0 | 0 |
| 40 | sm_86 shared/atomic u8→u64 local=u64 t256 i16 r1 cap=48KiB | scalar | 52 | 0/0 | 408 | +0 | 16 | 0 | 0 | 0 | 16 | 16 | 107 | 40 | 0 | 0 | 0 | 0 |
| 41 | sm_86 shared/atomic u8→u64 local=u64 t256 i16 r1 cap=48KiB | full_tile | 51 | 0/0 | 584 | +176 | 32 | 0 | 0 | 0 | 32 | 32 | 142 | 58 | 0 | 0 | 0 | 0 |
| 42 | sm_86 shared/atomic u8→u64 local=u64 t256 i16 r1 cap=48KiB | vector4 | 57 | 0/0 | 1120 | +712 | 48 | 4 | 0 | 0 | 64 | 64 | 264 | 111 | 0 | 0 | 0 | 0 |

PTXAS log: `/home/b/gpu_histogram/results/a5000-profiled/build.log`; SHA256 `d004bbf150bb4eeedde5dd8303fc000e388db7607ea67d7496457479260bf5bb`. Parsed 772 function records; 0 report nonzero spill stores or loads.

Exact function identities:

1. `void gh::(anonymous namespace)::shared_histogram<unsigned int, unsigned int, unsigned int, 256, 8, 1, (gh::(anonymous namespace)::Update)0, false>(unsigned int const*, unsigned long, unsigned int, unsigned int*, unsigned int*)`
2. `void gh::(anonymous namespace)::shared_histogram_loaded<unsigned int, unsigned int, unsigned int, 256, 8, 1, (gh::(anonymous namespace)::Update)0, false, (gh::LoadPolicy)1, 49152ul>(unsigned int const*, unsigned long, unsigned int, unsigned int*, unsigned int*)`
3. `void gh::(anonymous namespace)::shared_histogram_loaded<unsigned int, unsigned int, unsigned int, 256, 8, 1, (gh::(anonymous namespace)::Update)0, false, (gh::LoadPolicy)2, 49152ul>(unsigned int const*, unsigned long, unsigned int, unsigned int*, unsigned int*)`
4. `void gh::(anonymous namespace)::shared_histogram_loaded<unsigned int, unsigned int, unsigned int, 256, 8, 1, (gh::(anonymous namespace)::Update)0, false, (gh::LoadPolicy)2, 98304ul>(unsigned int const*, unsigned long, unsigned int, unsigned int*, unsigned int*)`
5. `void gh::(anonymous namespace)::shared_histogram<unsigned int, unsigned int, unsigned int, 256, 16, 1, (gh::(anonymous namespace)::Update)0, false>(unsigned int const*, unsigned long, unsigned int, unsigned int*, unsigned int*)`
6. `void gh::(anonymous namespace)::shared_histogram_loaded<unsigned int, unsigned int, unsigned int, 256, 16, 1, (gh::(anonymous namespace)::Update)0, false, (gh::LoadPolicy)1, 49152ul>(unsigned int const*, unsigned long, unsigned int, unsigned int*, unsigned int*)`
7. `void gh::(anonymous namespace)::shared_histogram_loaded<unsigned int, unsigned int, unsigned int, 256, 16, 1, (gh::(anonymous namespace)::Update)0, false, (gh::LoadPolicy)2, 49152ul>(unsigned int const*, unsigned long, unsigned int, unsigned int*, unsigned int*)`
8. `void gh::(anonymous namespace)::shared_histogram<unsigned int, unsigned long long, unsigned int, 256, 8, 1, (gh::(anonymous namespace)::Update)0, false>(unsigned int const*, unsigned long, unsigned int, unsigned long long*, unsigned int*)`
9. `void gh::(anonymous namespace)::shared_histogram_loaded<unsigned int, unsigned long long, unsigned int, 256, 8, 1, (gh::(anonymous namespace)::Update)0, false, (gh::LoadPolicy)1, 49152ul>(unsigned int const*, unsigned long, unsigned int, unsigned long long*, unsigned int*)`
10. `void gh::(anonymous namespace)::shared_histogram_loaded<unsigned int, unsigned long long, unsigned int, 256, 8, 1, (gh::(anonymous namespace)::Update)0, false, (gh::LoadPolicy)2, 49152ul>(unsigned int const*, unsigned long, unsigned int, unsigned long long*, unsigned int*)`
11. `void gh::(anonymous namespace)::shared_histogram_loaded<unsigned int, unsigned long long, unsigned int, 256, 8, 1, (gh::(anonymous namespace)::Update)0, false, (gh::LoadPolicy)2, 98304ul>(unsigned int const*, unsigned long, unsigned int, unsigned long long*, unsigned int*)`
12. `void gh::(anonymous namespace)::shared_histogram<unsigned int, unsigned long long, unsigned int, 256, 16, 1, (gh::(anonymous namespace)::Update)0, false>(unsigned int const*, unsigned long, unsigned int, unsigned long long*, unsigned int*)`
13. `void gh::(anonymous namespace)::shared_histogram_loaded<unsigned int, unsigned long long, unsigned int, 256, 16, 1, (gh::(anonymous namespace)::Update)0, false, (gh::LoadPolicy)1, 49152ul>(unsigned int const*, unsigned long, unsigned int, unsigned long long*, unsigned int*)`
14. `void gh::(anonymous namespace)::shared_histogram_loaded<unsigned int, unsigned long long, unsigned int, 256, 16, 1, (gh::(anonymous namespace)::Update)0, false, (gh::LoadPolicy)2, 49152ul>(unsigned int const*, unsigned long, unsigned int, unsigned long long*, unsigned int*)`
15. `void gh::(anonymous namespace)::shared_histogram<unsigned int, unsigned long long, unsigned long long, 256, 8, 1, (gh::(anonymous namespace)::Update)0, false>(unsigned int const*, unsigned long, unsigned int, unsigned long long*, unsigned long long*)`
16. `void gh::(anonymous namespace)::shared_histogram_loaded<unsigned int, unsigned long long, unsigned long long, 256, 8, 1, (gh::(anonymous namespace)::Update)0, false, (gh::LoadPolicy)1, 49152ul>(unsigned int const*, unsigned long, unsigned int, unsigned long long*, unsigned long long*)`
17. `void gh::(anonymous namespace)::shared_histogram_loaded<unsigned int, unsigned long long, unsigned long long, 256, 8, 1, (gh::(anonymous namespace)::Update)0, false, (gh::LoadPolicy)2, 49152ul>(unsigned int const*, unsigned long, unsigned int, unsigned long long*, unsigned long long*)`
18. `void gh::(anonymous namespace)::shared_histogram_loaded<unsigned int, unsigned long long, unsigned long long, 256, 8, 1, (gh::(anonymous namespace)::Update)0, false, (gh::LoadPolicy)2, 98304ul>(unsigned int const*, unsigned long, unsigned int, unsigned long long*, unsigned long long*)`
19. `void gh::(anonymous namespace)::shared_histogram<unsigned int, unsigned long long, unsigned long long, 256, 16, 1, (gh::(anonymous namespace)::Update)0, false>(unsigned int const*, unsigned long, unsigned int, unsigned long long*, unsigned long long*)`
20. `void gh::(anonymous namespace)::shared_histogram_loaded<unsigned int, unsigned long long, unsigned long long, 256, 16, 1, (gh::(anonymous namespace)::Update)0, false, (gh::LoadPolicy)1, 49152ul>(unsigned int const*, unsigned long, unsigned int, unsigned long long*, unsigned long long*)`
21. `void gh::(anonymous namespace)::shared_histogram_loaded<unsigned int, unsigned long long, unsigned long long, 256, 16, 1, (gh::(anonymous namespace)::Update)0, false, (gh::LoadPolicy)2, 49152ul>(unsigned int const*, unsigned long, unsigned int, unsigned long long*, unsigned long long*)`
22. `void gh::(anonymous namespace)::shared_histogram<unsigned char, unsigned int, unsigned int, 256, 8, 1, (gh::(anonymous namespace)::Update)0, false>(unsigned char const*, unsigned long, unsigned int, unsigned int*, unsigned int*)`
23. `void gh::(anonymous namespace)::shared_histogram_loaded<unsigned char, unsigned int, unsigned int, 256, 8, 1, (gh::(anonymous namespace)::Update)0, false, (gh::LoadPolicy)1, 49152ul>(unsigned char const*, unsigned long, unsigned int, unsigned int*, unsigned int*)`
24. `void gh::(anonymous namespace)::shared_histogram_loaded<unsigned char, unsigned int, unsigned int, 256, 8, 1, (gh::(anonymous namespace)::Update)0, false, (gh::LoadPolicy)2, 49152ul>(unsigned char const*, unsigned long, unsigned int, unsigned int*, unsigned int*)`
25. `void gh::(anonymous namespace)::shared_histogram_loaded<unsigned char, unsigned int, unsigned int, 256, 8, 1, (gh::(anonymous namespace)::Update)0, false, (gh::LoadPolicy)2, 98304ul>(unsigned char const*, unsigned long, unsigned int, unsigned int*, unsigned int*)`
26. `void gh::(anonymous namespace)::shared_histogram<unsigned char, unsigned int, unsigned int, 256, 16, 1, (gh::(anonymous namespace)::Update)0, false>(unsigned char const*, unsigned long, unsigned int, unsigned int*, unsigned int*)`
27. `void gh::(anonymous namespace)::shared_histogram_loaded<unsigned char, unsigned int, unsigned int, 256, 16, 1, (gh::(anonymous namespace)::Update)0, false, (gh::LoadPolicy)1, 49152ul>(unsigned char const*, unsigned long, unsigned int, unsigned int*, unsigned int*)`
28. `void gh::(anonymous namespace)::shared_histogram_loaded<unsigned char, unsigned int, unsigned int, 256, 16, 1, (gh::(anonymous namespace)::Update)0, false, (gh::LoadPolicy)2, 49152ul>(unsigned char const*, unsigned long, unsigned int, unsigned int*, unsigned int*)`
29. `void gh::(anonymous namespace)::shared_histogram<unsigned char, unsigned long long, unsigned int, 256, 8, 1, (gh::(anonymous namespace)::Update)0, false>(unsigned char const*, unsigned long, unsigned int, unsigned long long*, unsigned int*)`
30. `void gh::(anonymous namespace)::shared_histogram_loaded<unsigned char, unsigned long long, unsigned int, 256, 8, 1, (gh::(anonymous namespace)::Update)0, false, (gh::LoadPolicy)1, 49152ul>(unsigned char const*, unsigned long, unsigned int, unsigned long long*, unsigned int*)`
31. `void gh::(anonymous namespace)::shared_histogram_loaded<unsigned char, unsigned long long, unsigned int, 256, 8, 1, (gh::(anonymous namespace)::Update)0, false, (gh::LoadPolicy)2, 49152ul>(unsigned char const*, unsigned long, unsigned int, unsigned long long*, unsigned int*)`
32. `void gh::(anonymous namespace)::shared_histogram_loaded<unsigned char, unsigned long long, unsigned int, 256, 8, 1, (gh::(anonymous namespace)::Update)0, false, (gh::LoadPolicy)2, 98304ul>(unsigned char const*, unsigned long, unsigned int, unsigned long long*, unsigned int*)`
33. `void gh::(anonymous namespace)::shared_histogram<unsigned char, unsigned long long, unsigned int, 256, 16, 1, (gh::(anonymous namespace)::Update)0, false>(unsigned char const*, unsigned long, unsigned int, unsigned long long*, unsigned int*)`
34. `void gh::(anonymous namespace)::shared_histogram_loaded<unsigned char, unsigned long long, unsigned int, 256, 16, 1, (gh::(anonymous namespace)::Update)0, false, (gh::LoadPolicy)1, 49152ul>(unsigned char const*, unsigned long, unsigned int, unsigned long long*, unsigned int*)`
35. `void gh::(anonymous namespace)::shared_histogram_loaded<unsigned char, unsigned long long, unsigned int, 256, 16, 1, (gh::(anonymous namespace)::Update)0, false, (gh::LoadPolicy)2, 49152ul>(unsigned char const*, unsigned long, unsigned int, unsigned long long*, unsigned int*)`
36. `void gh::(anonymous namespace)::shared_histogram<unsigned char, unsigned long long, unsigned long long, 256, 8, 1, (gh::(anonymous namespace)::Update)0, false>(unsigned char const*, unsigned long, unsigned int, unsigned long long*, unsigned long long*)`
37. `void gh::(anonymous namespace)::shared_histogram_loaded<unsigned char, unsigned long long, unsigned long long, 256, 8, 1, (gh::(anonymous namespace)::Update)0, false, (gh::LoadPolicy)1, 49152ul>(unsigned char const*, unsigned long, unsigned int, unsigned long long*, unsigned long long*)`
38. `void gh::(anonymous namespace)::shared_histogram_loaded<unsigned char, unsigned long long, unsigned long long, 256, 8, 1, (gh::(anonymous namespace)::Update)0, false, (gh::LoadPolicy)2, 49152ul>(unsigned char const*, unsigned long, unsigned int, unsigned long long*, unsigned long long*)`
39. `void gh::(anonymous namespace)::shared_histogram_loaded<unsigned char, unsigned long long, unsigned long long, 256, 8, 1, (gh::(anonymous namespace)::Update)0, false, (gh::LoadPolicy)2, 98304ul>(unsigned char const*, unsigned long, unsigned int, unsigned long long*, unsigned long long*)`
40. `void gh::(anonymous namespace)::shared_histogram<unsigned char, unsigned long long, unsigned long long, 256, 16, 1, (gh::(anonymous namespace)::Update)0, false>(unsigned char const*, unsigned long, unsigned int, unsigned long long*, unsigned long long*)`
41. `void gh::(anonymous namespace)::shared_histogram_loaded<unsigned char, unsigned long long, unsigned long long, 256, 16, 1, (gh::(anonymous namespace)::Update)0, false, (gh::LoadPolicy)1, 49152ul>(unsigned char const*, unsigned long, unsigned int, unsigned long long*, unsigned long long*)`
42. `void gh::(anonymous namespace)::shared_histogram_loaded<unsigned char, unsigned long long, unsigned long long, 256, 16, 1, (gh::(anonymous namespace)::Update)0, false, (gh::LoadPolicy)2, 49152ul>(unsigned char const*, unsigned long, unsigned int, unsigned long long*, unsigned long long*)`
