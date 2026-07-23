# Teacher-Correction Transfer Report

Complete seeds: 4/4.
Group A counts by seed: [18, 10, 9, 7].
Group A denotes cases misclassified by the CT-text teacher and correctly classified by the CT-text-CNV teacher.
Student comparisons use each student's validation-selected threshold; primary teacher grouping uses threshold 0.5.
The analysis asks whether students use additional full-modality teacher supervision; it does not demonstrate that the student learned genomic representations.
Supervised-run overrides: seed44=/home/mmy/code/outputs0531_teacher_homogeneous_gene_test/densenet3d121_ct_text_teacher_strict_seed44_repeat.

## Primary descriptive findings

Across seeds, Group A contained 44 seed-case observations; these are not treated as independent patients.
The KD student was correct for 27/44 Group A observations, compared with 26/44 for the hard-label student.
KD-only correct observations: 3; supervised-only correct observations: 2.
The ensemble sensitivity grouping contained 13 Group A identities.
Cases for which the KD student was correct are interpreted as full-teacher corrections reflected in student behavior; incorrect KD cases are corrections not inherited under this operational definition.

## Confidence and interval assessment

Mean full-teacher confidence was 0.7999 when the KD student was correct and 0.8500 when it was incorrect.
The seed-averaged Group A accuracy difference (KD minus supervised) was 0.011111111111111113; the 95% bootstrap interval was [-0.10576923076923075, 0.12581699346405228].
Because the interval includes zero, the transfer comparison remains descriptive rather than interval-supported.
Confidence associations are descriptive and do not establish that confidence measures genomic knowledge transferability.

## Seed-level exact paired comparisons

- seed42: Group A n=18, KD-only correct=1, supervised-only correct=0, exact McNemar p=1.0000.
- seed43: Group A n=10, KD-only correct=1, supervised-only correct=0, exact McNemar p=1.0000.
- seed44: Group A n=9, KD-only correct=1, supervised-only correct=2, exact McNemar p=1.0000.
- seed45: Group A n=7, KD-only correct=0, supervised-only correct=0, exact McNemar p=1.0000.

