# Pepid: research-oriented next-generation peptide search engine

===

See upcoming paper for more details.

IMPORTANT: Pepid relies on numpy for some operations, which in turn relies on a platform BLAS implementation for efficiency. This may result, if running pepid in parallel, in exhausting resource limits (e.g. RLIMIT\_NPROC on linux).
This can be avoided by setting the right environment variables, at the likely cost of performance (for example: OMP\_NUM\_THREADS=1), or by reducing the amount of processes running in parallel.
