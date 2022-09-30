# Pepid: Research-Oriented, Highly Configurable Peptide Search Engine

===

See upcoming paper for more details.

Pepid is currently work in progress. Missing elements include stronger report generation, full port and proper integration of deep learning-driven components (spectrum property extraction, theoretical spectrum generation, and more), and an alternative to TDA-FDR evaluation.

An example config is provided in data/default.cfg: it should be copied somewhere and modified to suit user preferences (keys not filled in the user file are taken from the default config)

IMPORTANT: Pepid relies on numpy for some operations, which in turn relies on a platform BLAS implementation for efficiency. This may result, if running pepid in parallel, in exhausting resource limits (e.g. RLIMIT\_NPROC on linux).
This can be avoided by setting the right environment variables, at the likely cost of performance (for example: OMP\_NUM\_THREADS=1), or by reducing the amount of processes running in parallel.
