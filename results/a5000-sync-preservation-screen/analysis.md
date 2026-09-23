# Matched synchronization AA screen

Audited 64 processes, 2048 quartets, and 8192 raw positions.

Each row is a separate case/comparison/batch/mode stratum with two process repetitions. AA ratios are slot B / slot A of the same implementation. Values below are geometric means of process summaries; ranges are the two process values. These are descriptive screens, not confidence intervals.

| Case | AA binding | Batch | Sync mode | AA event ratio; process range | Event us/op; process range | Host us/quartet; process range |
|---|---|---:|---|---:|---:|---:|
| single | new-new | 32 | position | 1.017287; 1.000288–1.034576 | 5.932968; 5.524881–6.371198 | 2207.748603; 2096.372674–2325.041704 |
| single | new-new | 32 | quartet | 0.997238; 0.994480–1.000004 | 5.445411; 5.430869–5.459993 | 1923.757346; 1917.520082–1930.014898 |
| single | new-new | 64 | position | 1.023151; 0.999611–1.047244 | 5.984073; 5.983699–5.984448 | 3891.643373; 3862.256437–3921.253906 |
| single | new-new | 64 | quartet | 0.964222; 0.946399–0.982380 | 5.860799; 5.823641–5.898193 | 3502.846089; 3485.347115–3520.432920 |
| single | new-new | 128 | position | 0.981316; 0.977047–0.985605 | 5.854087; 5.847128–5.861053 | 6917.948076; 6775.103093–7063.804775 |
| single | new-new | 128 | quartet | 0.976143; 0.958302–0.994316 | 5.939274; 5.835630–6.044759 | 6729.061158; 6623.283632–6836.528010 |
| single | new-new | 256 | position | 0.973809; 0.950305–0.997896 | 5.874593; 5.774459–5.976464 | 13737.547261; 12850.137019–14686.240657 |
| single | new-new | 256 | quartet | 0.996407; 0.977429–1.015753 | 5.769470; 5.740884–5.798198 | 12691.792238; 12397.408696–12993.166084 |
| single | old-old | 32 | position | 1.000622; 1.000457–1.000787 | 6.148694; 5.465120–6.917770 | 2972.800821; 2141.194124–4127.390704 |
| single | old-old | 32 | quartet | 0.998503; 0.996451–1.000560 | 5.488848; 5.454934–5.522972 | 2015.122880; 1961.494003–2070.218015 |
| single | old-old | 64 | position | 1.022001; 1.006617–1.037621 | 5.876183; 5.855694–5.896743 | 3746.354645; 3689.293859–3804.297967 |
| single | old-old | 64 | quartet | 0.968055; 0.936811–1.000341 | 5.827154; 5.724465–5.931685 | 3577.854262; 3414.731575–3748.769365 |
| single | old-old | 128 | position | 1.002734; 0.996741–1.008763 | 5.789462; 5.780510–5.798429 | 6939.932237; 6883.830773–6996.490913 |
| single | old-old | 128 | quartet | 0.964155; 0.963787–0.964524 | 5.806133; 5.774599–5.837839 | 6587.605067; 6571.710687–6603.537888 |
| single | old-old | 256 | position | 1.002796; 0.988564–1.017234 | 5.718894; 5.642634–5.796185 | 13128.736417; 13019.682037–13238.704249 |
| single | old-old | 256 | quartet | 0.993574; 0.971787–1.015850 | 5.901010; 5.824564–5.978460 | 12956.976966; 12668.624121–13251.893062 |
| stream4096 | new-new | 32 | position | 1.003668; 0.965851–1.042965 | 25.070227; 23.624589–26.604326 | 4274.800243; 3807.332415–4799.664208 |
| stream4096 | new-new | 32 | quartet | 0.938829; 0.884021–0.997034 | 27.363873; 24.934869–30.029496 | 5228.240556; 3922.270864–6969.049374 |
| stream4096 | new-new | 64 | position | 0.978822; 0.944307–1.014600 | 26.340779; 25.642991–27.057555 | 8046.062146; 7789.675100–8310.887839 |
| stream4096 | new-new | 64 | quartet | 1.003961; 0.986635–1.021590 | 24.259233; 23.164661–25.405525 | 7067.142990; 6686.393369–7469.573996 |
| stream4096 | new-new | 128 | position | 0.964447; 0.910812–1.021240 | 25.446396; 23.669803–27.356336 | 14898.691066; 13411.736227–16550.504105 |
| stream4096 | new-new | 128 | quartet | 1.010389; 1.002781–1.018054 | 24.160678; 20.193120–28.907783 | 13570.923285; 11043.465572–16676.826456 |
| stream4096 | new-new | 256 | position | 1.040271; 1.022684–1.058160 | 24.886668; 23.913615–25.899315 | 27504.414994; 26206.215249–28866.924773 |
| stream4096 | new-new | 256 | quartet | 0.991171; 0.962212–1.021001 | 23.166540; 22.678885–23.664682 | 24930.550105; 24265.545129–25613.779755 |
| stream4096 | old-old | 32 | position | 0.983343; 0.956421–1.011022 | 23.671902; 22.427166–24.985723 | 3773.993393; 3511.937595–4055.603422 |
| stream4096 | old-old | 32 | quartet | 1.005620; 0.993708–1.017675 | 25.070143; 22.723721–27.658854 | 3860.337946; 3516.486496–4237.812110 |
| stream4096 | old-old | 64 | position | 0.944595; 0.915095–0.975046 | 24.866515; 19.911801–31.054125 | 7493.484564; 5808.898437–9666.602286 |
| stream4096 | old-old | 64 | quartet | 1.095725; 1.064134–1.128255 | 37.435795; 30.985891–45.228285 | 14641.531037; 9837.545539–21791.455019 |
| stream4096 | old-old | 128 | position | 0.944210; 0.924697–0.964134 | 25.185331; 22.790556–27.831744 | 14538.554242; 13019.943111–16234.292090 |
| stream4096 | old-old | 128 | quartet | 1.008452; 0.972059–1.046207 | 28.153990; 26.076628–30.396843 | 16155.993385; 14443.385592–18071.671672 |
| stream4096 | old-old | 256 | position | 1.007067; 0.986429–1.028137 | 24.461518; 24.089616–24.839161 | 26629.656012; 26224.085438–27041.498968 |
| stream4096 | old-old | 256 | quartet | 0.989730; 0.972848–1.006905 | 23.991012; 21.958701–26.211417 | 25579.338509; 23368.674214–27999.130483 |

## Metric definitions

- `event_us`: CUDA events around complete histogram batch, divided by batch; excludes pinned snapshot copy.
- `submit_us`: host submission of timed batch only, divided by batch; excludes snapshot submission.
- `total_host_us`: position mode only: timed batch enqueue through pinned snapshot completion, divided by batch; blank in quartet mode.
- `quartet_host_us`: one host interval from before first untimed launch to synchronization completing fourth pinned copy, in microseconds per quartet; repeated on four rows, never four independent samples.
- `timing_pair`: A first/second occurrence use IDs 0/1, B first/second use 2/3; four unique event pairs and graph instances per quartet.

## Limits

- AA screening only; no old/new performance conclusion, equivalence test, significance claim, or default promotion.
- Each case/comparison/batch/mode has two process repetitions. Never pool workloads, AA bindings, batches, or modes as replications of one effect.
- Pairs of modes run in adjacent distinct processes with reversed mode order on repetition two; GPU clocks remain unlocked.
- Both matched modes use four pinned snapshots and four occurrence-specific timing pairs/graphs. Compared with legacy, both change graph count and snapshot storage.
- Every quartet position is validated against CPU counts and canaries; no timing-position or outlier deletion.
- The position-mode quartet host interval includes CPU checks/event reads between positions. Both modes exclude validation after the fourth snapshot completion boundary.
- Each graph position has one preceding untimed graph batch; stream positions have one untimed operation. Host completion includes waiting for this work.
- Quartet mode cannot observe independent per-position host completion timestamps. It deliberately leaves total_host_us empty.
- Absolute latency and AA ratio variation are descriptive screens. Two repetitions cannot establish a precision plateau or zero loss.

Executable SHA-256: `009422daaf686c18abe453629eaeed56641494c0f5cd507348eaf4d7842f2350`.
