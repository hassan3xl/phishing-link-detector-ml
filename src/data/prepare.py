"""Phase 1: Data acquisition and cleaning.

Loads the raw PhiUSIIL dataset, inspects and cleans it, then performs the
train/test split BEFORE any feature engineering happens (the data-leakage
safeguard described in the README's "Data Leakage Prevention" section).

Only the raw `URL` and `label` columns are kept. PhiUSIIL ships 53 additional
precomputed feature columns (URLLength, IsHTTPS, NoOfSubDomain, ...); this
project deliberately drops all of them here so that every feature used later
is engineered independently in Phase 3 from the raw URL string, rather than
imported from someone else's feature-selection decisions.

Run: python -m src.data.prepare
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = Path("data/raw/phiusiil.csv")
TRAIN_PATH = Path("data/train.csv")
TEST_PATH = Path("data/test.csv")

RANDOM_STATE = 42  # fixed seed -> exact reproduction of the split on every run
TEST_SIZE = 0.20


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[load] raw shape: {df.shape}")
    return df


def inspect(df: pd.DataFrame) -> None:
    print("\n[inspect] dtypes of columns we keep:")
    print(df[["URL", "label"]].dtypes)
    print("\n[inspect] label value counts:")
    print(df["label"].value_counts())
    print("\n[inspect] nulls in URL/label:")
    print(df[["URL", "label"]].isnull().sum())
    print("\n[inspect] duplicate URL count:", df["URL"].duplicated().sum())


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only URL + label; drop nulls, empty/malformed URLs, and duplicates."""
    n0 = len(df)
    df = df[["URL", "label"]].copy()

    # Drop rows with missing URL or label.
    df = df.dropna(subset=["URL", "label"])
    n1 = len(df)

    # Drop empty / whitespace-only / implausibly short URL strings (e.g. "a",
    # "-") that can't represent a real URL and would only inject noise into
    # feature engineering (a length-4 floor comfortably admits the shortest
    # plausible real URL, e.g. "a.co").
    df["URL"] = df["URL"].astype(str).str.strip()
    df = df[df["URL"].str.len() >= 4]
    n2 = len(df)

    # Check for the same URL appearing with conflicting labels before
    # deduplicating - if the same URL is labeled both ways that's a label
    # quality issue worth surfacing, not silently resolving.
    conflicts = (
        df.groupby("URL")["label"].nunique().loc[lambda s: s > 1]
    )
    if len(conflicts) > 0:
        print(f"[clean] WARNING: {len(conflicts)} URLs have conflicting labels; dropping them entirely")
        df = df[~df["URL"].isin(conflicts.index)]
    n3 = len(df)

    # Drop exact duplicate URLs, keeping the first occurrence.
    df = df.drop_duplicates(subset=["URL"], keep="first")
    n4 = len(df)

    df = df.reset_index(drop=True)

    print("\n[clean] row counts through pipeline:")
    print(f"  raw:                          {n0}")
    print(f"  after dropna:                 {n1} (-{n0 - n1})")
    print(f"  after min-length filter:      {n2} (-{n1 - n2})")
    print(f"  after conflicting-label drop: {n3} (-{n2 - n3})")
    print(f"  after dedup:                  {n4} (-{n3 - n4})")
    return df


def split(df: pd.DataFrame):
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def main():
    df = load_raw()
    inspect(df)
    df = clean(df)

    print("\n[clean] final class balance:")
    print(df["label"].value_counts())
    print(df["label"].value_counts(normalize=True))

    train_df, test_df = split(df)

    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print(f"\n[split] train: {train_df.shape} saved to {TRAIN_PATH}")
    print(train_df["label"].value_counts(normalize=True))
    print(f"\n[split] test: {test_df.shape} saved to {TEST_PATH}")
    print(test_df["label"].value_counts(normalize=True))


if __name__ == "__main__":
    main()
