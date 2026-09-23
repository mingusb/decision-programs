"""Fresh serial timings for bounded export batching and revised preparation."""
from run_campaign import run


def wide_args(outputs):
    return ["--objective", "regression" if outputs == 129 else "binary",
            "--outputs", str(outputs), "--rows", "1024" if outputs == 4096 else "4096",
            "--test-rows", "64" if outputs == 4096 else "256", "--features", "16",
            "--rounds", "3" if outputs == 129 else "1" if outputs == 4096 else "2",
            "--depth", "2", "--bins", "32" if outputs == 129 else "16",
            "--histogram", "global", "--output-tile", "16"]


if __name__ == "__main__":
    scalar = ["--seed", "20260922601", "--rows", "65536", "--test-rows", "8192",
              "--features", "32", "--rounds", "10", "--depth", "5", "--bins", "64", "--histogram", "shared"]
    for suffix, modes in [("a", ("hybrid", "stream", "graph")), ("b", ("graph", "stream", "hybrid"))]:
        for mode in modes:
            extra = [] if mode == "hybrid" else ["--tree-execution", mode, "--tree-export-batch", "16"]
            run(f"optimized-scalar-{mode}-{suffix}", scalar + extra, mode == "hybrid")
    for outputs in (129, 1024):
        args = wide_args(outputs)
        for suffix, batches in [("a", (0, 16)), ("b", (16, 0))]:
            for batch in batches:
                run(f"optimized-{outputs}-graph-batch{batch}-{suffix}",
                    args + ["--tree-execution", "graph", "--tree-export-batch", str(batch)])
        run(f"optimized-{outputs}-graph-batch1", args + ["--tree-execution", "graph", "--tree-export-batch", "1"])
        run(f"optimized-{outputs}-stream-batch16", args + ["--tree-execution", "stream", "--tree-export-batch", "16"])
    args = wide_args(4096)
    for suffix, modes in [("a", ("hybrid", "graph")), ("b", ("graph", "hybrid"))]:
        for mode in modes:
            extra = [] if mode == "hybrid" else ["--tree-execution", "graph", "--tree-export-batch", "16"]
            run(f"optimized-4096-{mode}-{suffix}", args + extra, mode == "hybrid")
    # Fresh input seed confirms the preselected batch16 graph candidate.
    args = wide_args(129) + ["--seed", "20260922602"]
    run("confirm-129-hybrid", args, True)
    run("confirm-129-graph-batch16", args + ["--tree-execution", "graph", "--tree-export-batch", "16"])
    for rows, features in ((1048576, 8), (16777216, 1)):
        for policy in ("radix8", "radix4"):
            run(f"optimized-quantize-{rows}x{features}-{policy}",
                ["--rows", str(rows), "--test-rows", "128", "--features", str(features),
                 "--rounds", "0", "--bins", "256", "--histogram", "global", "--quantize-policy", policy])
