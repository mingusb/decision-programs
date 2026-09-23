# What the bounded oracle establishes

All five examined first-divergence nodes choose splits with the same optimal
integer sufficient statistics. Their positive gains range from 5.7944 to
509.6908; the next distinct count signature trails by 0.2566 to 9.7680. These
cases are mathematical ties between good splits, not near-zero split gains or
obviously inferior choices. The conclusion holds for the 80/120-digit sigmoid
reference and for exact rational sums of the separately identified host
binary64 derivative values. It does not establish the unrecorded CUDA sums.

The largest prediction change has a concrete explanation. Output 573's RR node
contains 226 training rows, including 193 positives. Features 196, 262 and 290
each send 224 rows/193 positives left and two negative rows right. Their
mathematical gains and leaf values tie exactly. The selected left training
memberships differ by four rows, so equal sufficient statistics do not mean
the feature tests are interchangeable on unseen rows.

Validation row 526 has feature290=1, but feature262=0 and feature196=0. The
original baseline therefore reaches the right leaf, while both the candidate
and independent baseline repeat reach the left leaf. The exported increments
are -0.0036925094163568215 and +3.62287290776729 respectively. The latter is
already multiplied by learning_rate=0.1; clipping was disabled. The initial
probability is about 0.0191563. Applying the saved increments explains the
observed probabilities 0.019087089828201544 and 0.4224055565205715, including
their 0.4033184666923699 difference. Independent CPU transformation differs
from the stored GPU values by at most one binary64 ULP on these traces.

Output 87's selected root features 80, 114 and 407 even partition the same
training row IDs. Eleven features share the optimal counts. Output 8's two
co-optimal features share counts but change 174 training-row assignments.
Consequently, both redundant features and different feature tests with equal
class-count summaries occur in these bounded examples.

The implementation uses unordered FP64 atomic additions for histogram
statistics. These observations are consistent with feature-specific rounding
breaking mathematical ties before the discrete feature-index tie rule is
reached. This is an inference: the saved models contain neither the actual
histogram inputs nor the losing GPU candidate scores. A captured real-node
histogram and candidate replay would be needed to locate that mechanism
directly. Nothing here overturns any strict quality-gate failure.

An optional future design is integer count-derived statistics for the first
independent-logistic round, where each output has one margin and hence only
two gradient values and a constant Hessian. Exact counts would remove atomic
floating-point accumulation from that bounded case and connect directly to
our counting work. Computing G/H from counts changes floating-point summation,
can change which tied feature wins, and requires a separately declared
numerical contract plus full quality/performance experiments. It does not
automatically apply to later rounds with varying row margins. No such
production change was implemented in this task.
