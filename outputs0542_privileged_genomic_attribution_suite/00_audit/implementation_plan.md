# Implementation Plan

1. Reuse the existing cached-logits student trainer and add explicit attribution modes while preserving legacy option semantics.
2. Reuse each teacher cache identically across uniform and confidence arms.
3. Add a deterministic training-only weight permutation for shuffled confidence.
4. Materialize split-local, whole-row CNV permutations before teacher training and preserve target-to-donor mappings.
5. Save all validation-selected checkpoints atomically for new student runs; keep composite as primary.
6. Analyze effects with common-identity paired bootstrap inside seed, followed by seed-level averaging.
7. Reuse existing predictions for teacher-correction and triclass confusion analyses; never start training in analysis stages.
