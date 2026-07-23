# Known Limitations

1. The workstation has no PyTorch/torchvision environment, and outbound package sources
   were unreachable. Real 1-2 epoch training smoke is therefore environment-blocked here.
2. The synchronized local T0/T1 and R3 result copies contain metrics/predictions/caches but
   no model checkpoint binaries. Formal checkpoint-dependent runs must use the server copy.
3. The locked paper reference is not training-matched to R3: batch size, learning rate,
   checkpoint criterion, and mixed precision differ. `S0_MATCHED` is therefore required.
4. Original and repeat seed-44 teacher artifacts coexist. This suite follows the original
   seed-44 artifact referenced by the R3 cache metadata and does not merge repeat results.
5. TRI-S0 has 36 extra test identities per seed relative to TRI-T/TRI-SKD. Direct confusion
   comparisons use the 227 common identities and report the exclusions.
6. Teacher-correction Group A is small (18, 10, 9, and 7 cases by seed); its interval crosses
   zero (seed-mean accuracy difference 0.0111, 95% interval [-0.1058, 0.1258]) and the result
   remains descriptive. The logical seed-44 supervised reference uses the locked repeat run.
7. The factorial, shuffled-confidence, CNV-permutation, and checkpoint sensitivity formal
   results do not exist until the 32 training jobs are run.
