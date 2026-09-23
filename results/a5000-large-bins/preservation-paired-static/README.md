# Paired-build static audit

Audit passed. Old/current namespaces retain 751/775 device functions. See [comparison.json](comparison.json) for exact source-binary comparisons and hashes.

Both host adapters target their own namespaced histogram function, and both backends share one dynamic CUDA runtime. No NVIDIA benchmark histogram symbols were found.

Full instruction/control words are compared for the two flagged counting kernels and u32 clear, with resource records checked for every custom device function. These checks establish static preservation and backend isolation only; they do not establish equal runtime performance. No GPU queries or launches occurred.
