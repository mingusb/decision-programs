# Strict quality audit

Status: **regression**; allowance remains zero.

| Candidate | Gate | Regressed metrics | Max prediction difference | Decision changes | All-bin margin bound |
|---|---|---:|---:|---:|---:|
| scalar-split-a | pass | 0 | 2.2204460492503131e-16 | None | 4.3021142204224816e-16 |
| scalar-root-a | regression | 2 | 2.2204460492503131e-16 | None | 4.3021142204224816e-16 |
| scalar-both-a | pass | 0 | 2.2204460492503131e-16 | None | 3.8163916471489756e-16 |
| scalar-split-b | pass | 0 | 2.2204460492503131e-16 | None | 3.9551695252271202e-16 |
| scalar-root-b | pass | 0 | 2.2204460492503131e-16 | None | 4.0939474033052647e-16 |
| scalar-both-b | pass | 0 | 2.2204460492503131e-16 | None | 3.9551695252271202e-16 |
| 129-split-a | regression | 36 | 3.6082248300317588e-16 | None | 5.7245874707234634e-16 |
| 129-root-a | regression | 28 | 3.8857805861880479e-16 | None | 4.649058915617843e-16 |
| 129-both-a | regression | 27 | 4.9960036108132044e-16 | None | 5.9847959921199845e-16 |
| 129-split-b | regression | 35 | 3.6082248300317588e-16 | None | 5.8980598183211441e-16 |
| 129-root-b | regression | 33 | 5.2735593669694936e-16 | None | 5.7853027923826517e-16 |
| 129-both-b | regression | 36 | 5.5511151231257827e-16 | None | 5.620504062164855e-16 |
| 1024-split-a | regression | 258 | 2.2204460492503131e-16 | 0 | 1.2177758801357186e-15 |
| 1024-root-a | regression | 249 | 2.2204460492503131e-16 | 0 | 1.717376241217039e-15 |
| 1024-both-a | regression | 247 | 2.7755575615628914e-16 | 0 | 1.0685896612017132e-15 |
| 1024-split-b | regression | 211 | 2.7755575615628914e-16 | 0 | 1.1726730697603216e-15 |
| 1024-root-b | regression | 254 | 2.7755575615628914e-16 | 0 | 1.3322676295501878e-15 |
| 1024-both-b | regression | 246 | 2.2204460492503131e-16 | 0 | 1.1518563880485999e-15 |
| 4096-split-a | regression | 481 | 2.2204460492503131e-16 | 0 | 8.5348395018058909e-16 |
| 4096-root-a | regression | 490 | 2.2204460492503131e-16 | 0 | 8.9511731360403246e-16 |
| 4096-both-a | regression | 455 | 2.2204460492503131e-16 | 0 | 9.0899510141184692e-16 |
| 4096-split-b | regression | 457 | 2.2204460492503131e-16 | 0 | 8.3266726846886741e-16 |
| 4096-root-b | regression | 415 | 2.2204460492503131e-16 | 0 | 8.0838113980519211e-16 |
| 4096-both-b | regression | 423 | 2.2204460492503131e-16 | 0 | 8.4654505627668186e-16 |
| confirm-129-split | regression | 30 | 3.3306690738754696e-16 | None | 6.8695049648681561e-16 |
| confirm-129-root | regression | 24 | 4.7184478546569153e-16 | None | 4.2251358661760108e-16 |
| confirm-129-both | regression | 35 | 3.3306690738754696e-16 | None | 5.134781488891349e-16 |
| multiclass17-both | pass | 0 | 1.1102230246251565e-16 | 0 | 1.4432899320127035e-15 |

All-bin bounds use exact-real sums of stored leaves over intersecting encoded leaf regions. They exclude runtime arithmetic rounding. Measured quality failures remain failures.
