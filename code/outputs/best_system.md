# Best-system improvement ladder

Each step adds one technique on top of the previous; deltas are vs. the CodeBERT baseline (S0). Threshold-independent PR-AUC / ROC-AUC are the primary evidence of a genuinely stronger detector.


## devign: improvement ladder (test set)

| step | config | accuracy | balanced_accuracy | f1 | mcc | roc_auc | pr_auc |
|---|---|---|---|---|---|---|---|
| S0 | CodeBERT baseline | 0.6447 | nan | 0.5858 | 0.2846 | 0.7183 | 0.7093 |
| S1 | + structure backbone (UniXcoder) | 0.6597 (+0.0151) | nan (+nan) | 0.6566 (+0.0708) | 0.3285 (+0.0440) | 0.7456 (+0.0273) | 0.7392 (+0.0299) |
| S2 | + best imbalance [UniXcoder + focal loss] | 0.6403 (-0.0044) | 0.6490 (+nan) | 0.6544 (+0.0686) | 0.3013 (+0.0167) | 0.7327 (+0.0144) | 0.7279 (+0.0186) |
| S3 | + val-tuned threshold (mcc) | 0.6328 (-0.0118) | 0.5986 (+nan) | 0.3435 (-0.2423) | 0.3221 (+0.0375) | 0.7326 (+0.0142) | 0.7278 (+0.0185) |
| S4 | + 3-seed ensemble (final) | 0.6418 (-0.0028) | 0.6513 (+nan) | 0.6588 (+0.0730) | 0.3069 (+0.0224) | 0.7403 (+0.0219) | 0.7357 (+0.0264) |

## diversevul: improvement ladder (test set)

| step | config | accuracy | balanced_accuracy | f1 | mcc | roc_auc | pr_auc |
|---|---|---|---|---|---|---|---|
| S0 | CodeBERT baseline | 0.8092 | nan | 0.2255 | 0.1614 | 0.7128 | 0.1576 |
| S1 | + structure backbone (UniXcoder) | 0.7898 (-0.0193) | nan (+nan) | 0.2449 (+0.0194) | 0.1912 (+0.0298) | 0.7300 (+0.0172) | 0.1751 (+0.0175) |
| S2 | + best imbalance [UniXcoder + focal loss] | 0.7779 (-0.0313) | 0.6601 (+nan) | 0.2452 (+0.0198) | 0.1944 (+0.0330) | 0.7317 (+0.0190) | 0.1775 (+0.0199) |
| S3 | + val-tuned threshold (mcc) | 0.7982 (-0.0109) | 0.6529 (+nan) | 0.2490 (+0.0236) | 0.1946 (+0.0331) | 0.7319 (+0.0191) | 0.1777 (+0.0201) |
| S4 | + 3-seed ensemble (final) | 0.7874 (-0.0218) | 0.6638 (+nan) | 0.2520 (+0.0265) | 0.2017 (+0.0403) | 0.7386 (+0.0258) | 0.1852 (+0.0275) |
