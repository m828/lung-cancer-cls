# Triclass Confusion Analysis Report

- seed42: 227 common test identities; model-specific identities excluded before comparison: {'TRI_T': 0, 'TRI_S0': 36, 'TRI_SKD': 0}
- seed43: 227 common test identities; model-specific identities excluded before comparison: {'TRI_T': 0, 'TRI_S0': 36, 'TRI_SKD': 0}
- seed44: 227 common test identities; model-specific identities excluded before comparison: {'TRI_T': 0, 'TRI_S0': 36, 'TRI_SKD': 0}
- seed45: 227 common test identities; model-specific identities excluded before comparison: {'TRI_T': 0, 'TRI_S0': 36, 'TRI_SKD': 0}

TRI-T benign cases were most often assigned to `malignant` (mean row proportion 0.8068).
TRI-SKD malignant cases were most often assigned to `benign` (mean row proportion 0.5890).
The matrices show class-specific redistribution across teacher and students. They support a descriptive statement that teacher class capabilities were not reproduced identically by the distilled student.
No causal attribution to CNV is made because the comparison does not isolate CNV information from branch capacity and other teacher differences.
