# Resident trainer validation experiment

This validation follows [the algorithm decisions](ALGORITHM_DECISIONS.md).
The experiment removes host dependencies during tree growth while preserving
the existing scalar-tree objectives, exact feature cuts, split rules, and
generator-version-2 benchmark datasets. Stream submission and bounded graph
execution are candidates; neither is declared faster before end-to-end timing.

Independent CPU references validate weighted base scores and losses, gradients,
histograms, exhaustive small split candidates, routing, and predictions. GPU
initialization tests include uneven output counts, zero-weight rows, absent
classes, extreme finite targets, invalid labels even on zero-weight rows,
multiple reduction chunk counts, output canaries, and graph capture/replay.
Small deterministic training fixtures compare stream and graph tree structures,
leaf values, objective histories, and held-out predictions with explicit numeric
tolerances. These tolerances check correctness; they do not replace the strict
zero-allowance quality comparison used for benchmark promotion.

Instrumentation checks distinguish complete tree-build scopes from per-level
work, reject per-level host download scopes, and ensure terminal leaves need
no additional histogram level. Output tiling and deep constant-target cases
exercise memory bounds without requiring a large timing workload. Report both
resident training payload and feature-preparation peak; the two storage phases
must not be conflated.

The benchmark only adds execution-policy selection and metadata. Its generator,
weights, held-out seeds, targets, and CSV serialization remain unchanged so the
frozen hybrid executable can be compared on identical inputs. Root runs GPU
checks and timings serially, retains failed quality gates, and uses Nsight
separately from uninstrumented rankings. Synthetic correctness fixtures are not
evidence of NLP accuracy or representative high-dimensional throughput.
