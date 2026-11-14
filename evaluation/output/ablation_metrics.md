# End-to-end ablation, qwen2.5-7b, P2 balanced test split

## movies (n = 1749, generation errors = 0, raw threshold = 0.8222)

| Decision | Retrieval | Accuracy | Refusal rate | Retrieval rate |
|---|---|---|---|---|
| raw | off | 0.354 | 0.572 | - |
| calibrated | off | 0.434 | 0.544 | - |
| raw | on | 0.926 | - | 0.572 |
| calibrated | on | 0.978 | - | 0.544 |

Coverage-accuracy (accuracy of directly answered questions):

| Score | cov 0.2 | cov 0.4 | cov 0.6 | cov 0.8 | cov 1.0 |
|---|---|---|---|---|---|
| raw c | 0.963 | 0.841 | 0.703 | 0.555 | 0.449 |
| calibrated | 0.986 | 0.973 | 0.745 | 0.559 | 0.449 |

## songs (n = 148, generation errors = 0, raw threshold = 0.8636)

| Decision | Retrieval | Accuracy | Refusal rate | Retrieval rate |
|---|---|---|---|---|
| raw | off | 0.351 | 0.622 | - |
| calibrated | off | 0.392 | 0.588 | - |
| raw | on | 0.973 | - | 0.622 |
| calibrated | on | 0.980 | - | 0.588 |

Coverage-accuracy (accuracy of directly answered questions):

| Score | cov 0.2 | cov 0.4 | cov 0.6 | cov 0.8 | cov 1.0 |
|---|---|---|---|---|---|
| raw c | 0.967 | 0.883 | 0.708 | 0.563 | 0.453 |
| calibrated | 1.000 | 0.950 | 0.753 | 0.563 | 0.453 |

## basketball (n = 563, generation errors = 0, raw threshold = 0.8185)

| Decision | Retrieval | Accuracy | Refusal rate | Retrieval rate |
|---|---|---|---|---|
| raw | off | 0.181 | 0.702 | - |
| calibrated | off | 0.242 | 0.661 | - |
| raw | on | 0.883 | - | 0.702 |
| calibrated | on | 0.902 | - | 0.661 |

Coverage-accuracy (accuracy of directly answered questions):

| Score | cov 0.2 | cov 0.4 | cov 0.6 | cov 0.8 | cov 1.0 |
|---|---|---|---|---|---|
| raw c | 0.655 | 0.558 | 0.494 | 0.430 | 0.362 |
| calibrated | 0.770 | 0.659 | 0.553 | 0.441 | 0.362 |
