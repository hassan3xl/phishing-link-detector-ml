"""Applies `extract_features` to the train and test splits separately,
producing the feature matrices used by every later phase.

Run: python -m src.features.build_matrix
"""

from pathlib import Path

import pandas as pd

from src.features.extract import FEATURE_NAMES, extract_features

TRAIN_IN, TEST_IN = Path("data/train.csv"), Path("data/test.csv")
TRAIN_OUT, TEST_OUT = Path("data/train_features.csv"), Path("data/test_features.csv")

NEAR_ZERO_VAR_THRESHOLD = 0.01  # flag features whose variance is below this
HIGH_CORR_THRESHOLD = 0.90      # flag feature pairs above this |correlation|


def build(df: pd.DataFrame) -> pd.DataFrame:
    rows = [extract_features(u) for u in df["URL"]]
    X = pd.DataFrame(rows, columns=FEATURE_NAMES)
    X["label"] = df["label"].values
    return X


def check_variance(X: pd.DataFrame) -> None:
    variances = X[FEATURE_NAMES].var()
    low = variances[variances < NEAR_ZERO_VAR_THRESHOLD]
    print("\n[variance] feature variances (train):")
    print(variances.sort_values().to_string())
    if len(low) > 0:
        print(f"\n[variance] WARNING - near-zero variance (<{NEAR_ZERO_VAR_THRESHOLD}): {low.index.tolist()}")
    else:
        print(f"\n[variance] no features below the {NEAR_ZERO_VAR_THRESHOLD} threshold.")


def check_correlation(X: pd.DataFrame) -> None:
    corr = X[FEATURE_NAMES].corr()
    pairs = []
    for i, a in enumerate(FEATURE_NAMES):
        for b in FEATURE_NAMES[i + 1:]:
            c = corr.loc[a, b]
            if abs(c) >= HIGH_CORR_THRESHOLD:
                pairs.append((a, b, round(c, 3)))
    print(f"\n[correlation] pairs with |r| >= {HIGH_CORR_THRESHOLD}:")
    if pairs:
        for a, b, c in pairs:
            print(f"  {a} <-> {b}: r={c}")
    else:
        print("  none")


def main():
    train_df, test_df = pd.read_csv(TRAIN_IN), pd.read_csv(TEST_IN)

    X_train = build(train_df)
    X_test = build(test_df)

    check_variance(X_train)
    check_correlation(X_train)

    X_train.to_csv(TRAIN_OUT, index=False)
    X_test.to_csv(TEST_OUT, index=False)
    print(f"\n[save] {TRAIN_OUT} {X_train.shape}, {TEST_OUT} {X_test.shape}")


if __name__ == "__main__":
    main()
