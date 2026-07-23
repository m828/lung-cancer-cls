# Next-Step Recommendation

1. On the server, run the bounded smoke command and require all training/checkpoint modules
   to pass before allocating the four-seed jobs.
2. Run the 20-job binary factorial first. This directly identifies teacher-modality,
   confidence-weighting, final-vs-matched, and interaction effects.
3. Run the four shuffled-confidence jobs next; they are the smallest independent test of
   whether patient-specific confidence correspondence matters.
4. Run the eight permuted-CNV teacher/student jobs only after permutation manifests pass.
5. Execute checkpoint sensitivity and the two no-training analyses after all relevant files
   are complete. Do not select a new primary checkpoint from test performance.
6. Update the manuscript only after full summaries and paired intervals have been audited;
   this implementation turn intentionally leaves it unchanged.

