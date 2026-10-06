# Decision Programs paper

[Read the PDF](decision-programs-paper.pdf) · [Editable LuaLaTeX source](decision-programs-paper.tex)

**Compiling Boosted Trees into Compact, Exact Decision Programs** — Brian Mingus.

The manuscript includes the completed conversion and simplification results, executable decision equations, RL integration, scaling analysis, and the October 6, 2026 comparison of 0, 1, 2, 4 and 8 holdout confirmation stages. It has three pages of content plus references.

## Rebuild

With LuaLaTeX and the standard TeX packages used by the source installed:

```sh
cd docs/paper
lualatex -interaction=nonstopmode -halt-on-error decision-programs-paper.tex
lualatex -interaction=nonstopmode -halt-on-error decision-programs-paper.tex
```

The source embeds its ICML style and bibliography, and draws the figures with TikZ/PGFPlots. It does not require external figure files, datasets, or GPU computation to build. The embedded style retains its upstream notices; see [third-party notices](../../THIRD_PARTY_NOTICES.md).
