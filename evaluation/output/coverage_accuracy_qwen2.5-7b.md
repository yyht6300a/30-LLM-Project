# Coverage-accuracy, qwen2.5-7b, balanced test split

## movies (n_test = 1749)

| Score | cov 0.2 | cov 0.4 | cov 0.6 | cov 0.8 | cov 1.0 |
|---|---|---|---|---|---|
| off: raw c | 0.949 | 0.859 | 0.752 | 0.615 | 0.500 |
| on: MLP(full) | 0.991 | 0.976 | 0.830 | 0.623 | 0.500 |

## songs (n_test = 148)

| Score | cov 0.2 | cov 0.4 | cov 0.6 | cov 0.8 | cov 1.0 |
|---|---|---|---|---|---|
| off: raw c | 0.967 | 0.933 | 0.764 | 0.613 | 0.500 |
| on: MLP(full) | 1.000 | 0.950 | 0.831 | 0.622 | 0.500 |

## basketball (n_test = 563)

| Score | cov 0.2 | cov 0.4 | cov 0.6 | cov 0.8 | cov 1.0 |
|---|---|---|---|---|---|
| off: raw c | 0.566 | 0.619 | 0.604 | 0.557 | 0.499 |
| on: MLP(full) | 0.823 | 0.770 | 0.678 | 0.590 | 0.499 |
