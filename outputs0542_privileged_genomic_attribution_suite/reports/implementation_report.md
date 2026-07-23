# Implementation Report

## Material Passport

- Artifact: `0542-privileged-genomic-attribution-suite`
- Type: code experiment implementation and reproducibility audit
- Status: `IMPLEMENTED_TRAINING_SMOKE_PENDING_SERVER_ENVIRONMENT`
- Source commit: `613b86d8bee2b59e34bcd2cf844c2495fe7401b1` plus recorded working-tree source hashes
- Existing outputs: read-only
- Manuscript modified: no

## Required Questions

1. **Existing teacher checkpoints reused?** Existing T0/T1 predictions, metrics, manifests,
   and cached logits are reused. Local `.pt` checkpoint binaries are absent; formal runs must
   resolve them on the server. The permuted-CNV teacher is necessarily new.
2. **Missing teacher checkpoints?** All eight local T0/T1 `best_model.pt` files for seeds
   42-45 are absent from the synchronized workstation copy. Inventory paths are recorded.
3. **Locked baseline matched to R3?** No. Verified mismatches are batch size 4 vs 12,
   learning rate 3e-4 vs 1e-4, AUROC vs composite checkpoint selection, and disabled vs
   enabled mixed precision.
4. **Need `S0_MATCHED`?** Yes. It uses the R3 student architecture/training configuration
   with explicit KD-off semantics and preserves `S0_LOCKED_PRIMARY` as a separate concept.
5. **Five factorial arms runnable?** Their commands, cache reuse, output isolation, resume,
   and analyses pass syntax/dry-run checks. Real training smoke awaits a PyTorch server environment.
6. **Shuffled confidence changes only weights?** Yes. The training SampleID order, cached
   logits, labels, and data loader remain unchanged; a deterministic train-only derangement
   preserves the exact weight multiset. Utility and analysis audits pass.
7. **CNV permutation whole-row and split-local?** Yes. Train/val/test donor mappings are
   generated independently, complete feature rows are copied, labels are not inputs, fixed
   points are rejected, and source/donor/output hashes are stored.
8. **All checkpoint criteria saved?** New student runs atomically save validation loss,
   AUROC, F1, BAcc, composite, and last checkpoints. `best_model.pt` remains the validation
   composite primary model; test metrics never select a checkpoint. Resume checkpoints retain
   the historical primary best, all criterion-specific best records, history, and patience state.
9. **Teacher correction identity-paired?** Yes. Four seeds align by SampleID and label.
   Primary Group A sizes are 18/10/9/7. The seed-averaged KD-minus-supervised accuracy
   difference is 0.0111 with 95% bootstrap interval [-0.1058, 0.1258], so it is descriptive.
   The logical seed-44 supervised reference uses the locked `seed44_repeat` artifact.
10. **Triclass predictions complete?** TRI-T and TRI-SKD contain the expected 227 aligned
    test identities. TRI-S0 contains 36 additional identities per seed; comparison uses and
    reports the 227 common identities. Class mapping is verified as 0/1/2 = normal/benign/malignant.
11. **Unit tests?** 15 focused tests pass. One atomic PyTorch serialization test is skipped
    because PyTorch is unavailable locally; its code path remains in the server smoke plan.
12. **Smoke tests?** Static, dry-run, analysis, real teacher-correction, and real triclass
    smoke checks pass. Actual 1-2 epoch training smoke is explicitly blocked by the environment.
13. **Formal experiments pending?** All 32 formal training jobs remain unexecuted.
14. **Formal training job count?** 20 factorial students + 4 shuffled students + 4
    permuted-CNV teachers + 4 permuted-CNV students = 32.
15. **Leakage/identity/version conflicts?** T0/T1 manifests are byte-identical across each
    seed pair and no binary prediction alignment failure was found. The known triclass cohort
    difference and seed-44 original/repeat artifact distinction are explicitly recorded.
16. **Old defaults changed?** Legacy KD option semantics and numerical defaults are retained.
    Additive changes provide atomic writes, provenance, smoke bounds, explicit 0542 modes,
    and optional multi-checkpoint saving.
17. **Old outputs modified?** No.
18. **Manuscript modified?** No.

## Implemented Components

- Guarded factorial, shuffled-confidence, CNV-permutation, and suite launchers.
- Explicit KD-off/uniform/confidence/shuffled-confidence interface.
- Untempered confidence audit and temperature-scaled soft-target KD.
- Whole-row CNV permutation with donor manifests and hashes.
- Identity-safe paired bootstrap and factorial interaction analysis.
- Validation-only multi-checkpoint saving/evaluation.
- Resume-safe restoration of primary and criterion-specific validation state.
- Validation-selected operating-threshold evaluation for checkpoint sensitivity.
- Case-level teacher-correction and triclass confusion/profile analyses.
- Focused tests, analysis smoke fixtures, runbook, schemas, and formal commands.

## Verification Boundary

No claim is made about the five factorial effects, shuffled-confidence effect, CNV-permutation
effect, or checkpoint robustness until the formal jobs complete. Existing prediction analyses
are secondary descriptive outputs and are not evidence that the student learned genomic features.
