# Strict quality audit

Status: **regression**; allowance remains zero.

| Candidate | Gate | Regressed metrics | Max prediction difference | Decision changes | All-bin margin bound |
|---|---|---:|---:|---:|---:|
| scalar-counts-a | pass | 0 | 2.2204460492503131e-16 | None | 4.3021142204224816e-16 |
| scalar-split-a | pass | 0 | 2.2204460492503131e-16 | None | 4.1980308118638732e-16 |
| scalar-both-a | pass | 0 | 2.2204460492503131e-16 | None | 4.0592529337857286e-16 |
| scalar-deep-shared-a | pass | 0 | 2.2204460492503131e-16 | None | 4.3715031594615539e-16 |
| scalar-counts-b | pass | 0 | 2.2204460492503131e-16 | None | 3.8857805861880479e-16 |
| scalar-split-b | pass | 0 | 2.2204460492503131e-16 | None | 4.2327252813834093e-16 |
| scalar-both-b | pass | 0 | 2.2204460492503131e-16 | None | 3.677613769070831e-16 |
| scalar-deep-shared-b | pass | 0 | 2.2204460492503131e-16 | None | 4.163336342344337e-16 |
| 129-counts-a | regression | 41 | 5.5511151231257827e-16 | None | 6.9735883734267645e-16 |
| 129-split-a | regression | 27 | 5.134781488891349e-16 | None | 5.8980598183211441e-16 |
| 129-both-a | regression | 34 | 3.8857805861880479e-16 | None | 5.8286708792820718e-16 |
| 129-deep-shared-a | regression | 42 | 6.106226635438361e-16 | None | 6.6960326172704754e-16 |
| 129-counts-b | regression | 27 | 3.8857805861880479e-16 | None | 4.6837533851373792e-16 |
| 129-split-b | regression | 31 | 4.163336342344337e-16 | None | 4.5796699765787707e-16 |
| 129-both-b | regression | 39 | 4.163336342344337e-16 | None | 4.9960036108132044e-16 |
| 129-deep-shared-b | regression | 42 | 4.0245584642661925e-16 | None | 5.1694759584108851e-16 |
| 1024-counts-a | regression | 219 | 2.7755575615628914e-16 | 0 | 1.5022705301959149e-15 |
| 1024-split-a | regression | 248 | 2.7755575615628914e-16 | 0 | 1.5126788710517758e-15 |
| 1024-both-a | regression | 219 | 2.7755575615628914e-16 | 0 | 1.5057399771478686e-15 |
| 1024-deep-shared-a | regression | 269 | 3.3306690738754696e-16 | 0 | 1.4502288259166107e-15 |
| 1024-counts-b | regression | 240 | 3.3306690738754696e-16 | 0 | 1.519617764955683e-15 |
| 1024-split-b | regression | 269 | 3.3306690738754696e-16 | 0 | 1.3600232051658168e-15 |
| 1024-both-b | regression | 283 | 2.2204460492503131e-16 | 0 | 1.4988010832439613e-15 |
| 1024-deep-shared-b | regression | 270 | 2.2204460492503131e-16 | 0 | 1.3600232051658168e-15 |
| 4096-counts-a | regression | 474 | 2.2204460492503131e-16 | 0 | 9.7144514654701197e-16 |
| 4096-split-a | regression | 471 | 2.2204460492503131e-16 | 0 | 9.7144514654701197e-16 |
| 4096-both-a | regression | 473 | 2.2204460492503131e-16 | 0 | 8.8817841970012523e-16 |
| 4096-deep-shared-a | regression | 513 | 2.7755575615628914e-16 | 0 | 1.1657341758564144e-15 |
| 4096-counts-b | regression | 540 | 2.2204460492503131e-16 | 0 | 8.8817841970012523e-16 |
| 4096-split-b | regression | 454 | 2.2204460492503131e-16 | 0 | 8.3266726846886741e-16 |
| 4096-both-b | regression | 466 | 2.2204460492503131e-16 | 0 | 8.3266726846886741e-16 |
| 4096-deep-shared-b | regression | 505 | 2.2204460492503131e-16 | 0 | 9.9226182825873366e-16 |
| scalar-base-a | regression | 4 | 2.2204460492503131e-16 | None | 3.5735303605122226e-16 |
| scalar-base-b | regression | 2 | 2.2204460492503131e-16 | None | 3.5388358909926865e-16 |
| 129-counts-shared-a | regression | 31 | 3.8857805861880479e-16 | None | 5.7245874707234634e-16 |
| 129-counts-shared-b | regression | 27 | 4.8572257327350599e-16 | None | 5.6898930012039273e-16 |
| 4096-counts-shared-a | regression | 475 | 2.2204460492503131e-16 | 0 | 8.6042284408449632e-16 |
| 4096-counts-shared-b | regression | 454 | 2.2204460492503131e-16 | 0 | 7.7715611723760958e-16 |
| 129-base-b | regression | 43 | 6.3837823915946501e-16 | None | 7.0082828429463007e-16 |
| 129-base-a | regression | 37 | 6.3837823915946501e-16 | None | 7.0082828429463007e-16 |
| 129-base-c | regression | 32 | 5.5511151231257827e-16 | None | 6.0368376963992887e-16 |
| 129-base-a | regression | 27 | 5.5511151231257827e-16 | None | 6.0368376963992887e-16 |
| 129-base-d | regression | 40 | 3.7470027081099033e-16 | None | 6.4878658001532585e-16 |
| 129-base-a | regression | 26 | 3.7470027081099033e-16 | None | 6.4878658001532585e-16 |
| 1024-base-b | regression | 230 | 3.8857805861880479e-16 | 0 | 1.5681900222830336e-15 |
| 1024-base-a | regression | 280 | 3.8857805861880479e-16 | 0 | 1.5681900222830336e-15 |
| 1024-base-c | regression | 236 | 3.3306690738754696e-16 | 0 | 1.4849232954361469e-15 |
| 1024-base-a | regression | 258 | 3.3306690738754696e-16 | 0 | 1.4849232954361469e-15 |
| 1024-base-d | regression | 246 | 3.3306690738754696e-16 | 0 | 1.4432899320127035e-15 |
| 1024-base-a | regression | 259 | 3.3306690738754696e-16 | 0 | 1.4432899320127035e-15 |
| 4096-base-b | regression | 458 | 2.2204460492503131e-16 | 0 | 7.4940054162198066e-16 |
| 4096-base-a | regression | 470 | 2.2204460492503131e-16 | 0 | 7.4940054162198066e-16 |
| 4096-base-c | regression | 449 | 2.2204460492503131e-16 | 0 | 7.4593109467002705e-16 |
| 4096-base-a | regression | 459 | 2.2204460492503131e-16 | 0 | 7.4593109467002705e-16 |
| 4096-base-d | regression | 476 | 2.2204460492503131e-16 | 0 | 7.5633943552588789e-16 |
| 4096-base-a | regression | 486 | 2.2204460492503131e-16 | 0 | 7.5633943552588789e-16 |
| confirm-129-both | regression | 29 | 4.163336342344337e-16 | None | 5.6898930012039273e-16 |
| confirm-129-deep-shared | regression | 32 | 4.4408920985006262e-16 | None | 5.4123372450476381e-16 |
| confirm-4096-both | regression | 472 | 3.3306690738754696e-16 | 0 | 1.4432899320127035e-15 |
| confirm-4096-deep-shared | regression | 495 | 2.2204460492503131e-16 | 0 | 1.0824674490095276e-15 |
| multiclass17-both | pass | 0 | 8.3266726846886741e-17 | 0 | 1.0269562977782698e-15 |
| multiclass17-deep-shared | pass | 0 | 1.6653345369377348e-16 | 0 | 1.1657341758564144e-15 |

All-bin bounds use exact-real sums of stored leaves over intersecting encoded leaf regions. They exclude runtime arithmetic rounding. Measured quality failures remain failures.
