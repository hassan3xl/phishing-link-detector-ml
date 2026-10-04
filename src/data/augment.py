"""Augmentation pipeline for legitimate URLs to eliminate PhiUSIIL's zero-path bias.

Generates 50,000 diverse, realistic legitimate deep-linked URLs from existing
training domains (data/train.csv) spanning:
1. Web applications and session tokens (/chat/, /c/{uuid}, /workspace/{uuid})
2. Articles, wikis, and documentation (/wiki/{topic}, /docs/{guide})
3. Blog posts, news, and date-based paths (/news/2024/05/update.html)
4. Standard organizational subdirectories (/about-us, /pricing, /privacy-policy)
5. Search queries and paginated filters (?q=search+term&page=2)
6. Apex domain and subdomain diversity (with and without www.)

Extracts all 17 features and builds data/train_augmented_features.csv for model training.

Run: python -m src.data.augment
"""

import random
import uuid
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.features.extract import FEATURE_NAMES, extract_features

TRAIN_IN = Path("data/train.csv")
ORIGINAL_TRAIN_FEATURES = Path("data/train_features.csv")
AUGMENTED_FEATURES_OUT = Path("data/train_augmented_features.csv")

TEST_IN = Path("data/test.csv")
ORIGINAL_TEST_FEATURES = Path("data/test_features.csv")

NUM_AUGMENTED_SAMPLES = 50000
NUM_TEST_TARGET = 50000
RANDOM_STATE = 42

random.seed(RANDOM_STATE)

# Realistic topic slugs, documentation terms, and news categories
_TOPICS = [
    "Machine_learning", "Cybersecurity", "Artificial_intelligence", "Data_science",
    "Computer_network", "Web_development", "Python_programming", "Cloud_computing",
    "Open_source", "Software_engineering", "Cryptography", "Internet_protocol",
    "Database_management", "Operating_system", "Distributed_computing"
]

_SECTIONS = [
    "about-us", "contact", "privacy-policy", "terms-of-service", "pricing",
    "faq", "help-center", "team", "careers", "security", "features"
]

_DOCS = [
    "getting-started", "installation", "quickstart", "api-reference",
    "configuration", "architecture", "troubleshooting", "release-notes"
]

_SEARCH_QUERIES = [
    "machine+learning", "deep+learning", "cybersecurity+tools", "python+tutorial",
    "cloud+hosting", "best+practices", "documentation", "download+latest"
]


def generate_realistic_path() -> str:
    """Generates a realistic path pattern typical of modern web applications."""
    category = random.choices(
        ["webapp", "wiki", "docs", "news", "standard", "search", "repo"],
        weights=[0.20, 0.20, 0.15, 0.15, 0.15, 0.10, 0.05],
        k=1,
    )[0]

    if category == "webapp":
        # UUID and session-based deep links (e.g., ChatGPT, Claude, analytics dashboards)
        sub = random.choice(["c", "chat", "session", "workspace", "dashboard", "thread"])
        token = str(uuid.uuid4())
        return f"/{sub}/{token}"

    elif category == "wiki":
        # Wiki, knowledge base, and encyclopedic entries
        topic = random.choice(_TOPICS)
        return f"/wiki/{topic}"

    elif category == "docs":
        # Software documentation and user guides
        guide = random.choice(_DOCS)
        return f"/docs/{guide}"

    elif category == "news":
        # News articles and date-stamped blog posts
        year = random.choice(["2023", "2024", "2025"])
        month = f"{random.randint(1, 12):02d}"
        slug = random.choice(["latest-updates", "product-announcements", "quarterly-report", "research-insights"])
        ext = random.choice([".html", "", ".php"])
        return f"/news/{year}/{month}/{slug}{ext}"

    elif category == "search":
        # Query parameters and search pages
        q = random.choice(_SEARCH_QUERIES)
        page = random.randint(1, 5)
        return f"/search?q={q}&page={page}"

    elif category == "repo":
        # Developer repositories and project files
        user = random.choice(["torvalds", "apache", "google", "facebook", "microsoft"])
        repo = random.choice(["linux", "tensorflow", "react", "vscode", "spark"])
        return f"/{user}/{repo}"

    else:
        # Standard corporate pages
        section = random.choice(_SECTIONS)
        return f"/{section}"


def augment_training_set():
    print(f"[load] reading original training set from {TRAIN_IN}...")
    df_raw = pd.read_csv(TRAIN_IN)
    legit_urls = df_raw[df_raw["label"] == 1]["URL"].tolist()
    print(f"[info] found {len(legit_urls)} clean legitimate domains.")

    # Sample domains for augmentation
    sampled_domains = random.choices(legit_urls, k=NUM_AUGMENTED_SAMPLES)

    print(f"[augment] generating {NUM_AUGMENTED_SAMPLES} realistic deep-linked URLs...")
    augmented_rows = []

    for i, base_url in enumerate(tqdm(sampled_domains, desc="Generating train features")):
        # Alternate between apex domain (without www.) and standard (with www.)
        clean_base = base_url.rstrip("/")
        if i % 2 == 0 and "://www." in clean_base:
            clean_base = clean_base.replace("://www.", "://")

        path = generate_realistic_path()
        aug_url = clean_base + path

        f = extract_features(aug_url)
        f["label"] = 1  # Legitimate
        augmented_rows.append(f)

    df_aug = pd.DataFrame(augmented_rows)
    print(f"[info] generated augmented feature matrix with shape {df_aug.shape}")

    # Load original training feature matrix
    print(f"[load] loading original feature matrix from {ORIGINAL_TRAIN_FEATURES}...")
    df_orig = pd.read_csv(ORIGINAL_TRAIN_FEATURES)

    # Combine original + augmented
    df_combined = pd.concat([df_orig, df_aug[FEATURE_NAMES + ["label"]]], ignore_index=True)
    print(f"[combine] combined training set shape: {df_combined.shape}")
    print(f"[balance] class distribution in augmented train set:")
    print(df_combined["label"].value_counts(normalize=True))

    AUGMENTED_FEATURES_OUT.parent.mkdir(parents=True, exist_ok=True)
    df_combined.to_csv(AUGMENTED_FEATURES_OUT, index=False)
    print(f"[save] saved combined dataset to {AUGMENTED_FEATURES_OUT}")


def augment_test_set(target_count: int = NUM_TEST_TARGET):
    """Expands data/test.csv and data/test_features.csv to target_count (default: 50,000)

    by synthesizing realistic deep-linked URLs strictly from held-out test domains.
    Zero data leakage is preserved: only domains already inside data/test.csv are used.
    """
    print(f"\n[test] reading held-out test set from {TEST_IN}...")
    df_test_raw = pd.read_csv(TEST_IN)
    current_count = len(df_test_raw)
    needed = target_count - current_count

    if needed <= 0:
        print(f"[test] test set already has {current_count} samples (>= {target_count}). Nothing to add.")
        return

    legit_test_urls = df_test_raw[df_test_raw["label"] == 1]["URL"].tolist()
    print(f"[test] found {len(legit_test_urls)} legitimate test domains.")
    print(f"[test] generating {needed} realistic deep-linked URLs from test domains (strictly zero leakage)...")

    # Use independent random generator for test split
    rng = random.Random(RANDOM_STATE + 100)
    sampled_domains = rng.choices(legit_test_urls, k=needed)

    new_urls = []
    new_features = []

    for i, base_url in enumerate(tqdm(sampled_domains, desc="Generating test features")):
        clean_base = base_url.rstrip("/")
        if i % 2 == 0 and "://www." in clean_base:
            clean_base = clean_base.replace("://www.", "://")

        path = generate_realistic_path()
        aug_url = clean_base + path

        new_urls.append({"URL": aug_url, "label": 1})

        f = extract_features(aug_url)
        f["label"] = 1
        new_features.append(f)

    # 1. Update data/test.csv
    df_new_urls = pd.DataFrame(new_urls)
    df_test_combined = pd.concat([df_test_raw, df_new_urls], ignore_index=True)
    df_test_combined.to_csv(TEST_IN, index=False)
    print(f"[test] updated {TEST_IN}: {current_count} -> {len(df_test_combined)} rows.")

    # 2. Update data/test_features.csv
    df_orig_features = pd.read_csv(ORIGINAL_TEST_FEATURES)
    df_new_features = pd.DataFrame(new_features)
    df_features_combined = pd.concat([df_orig_features, df_new_features[FEATURE_NAMES + ["label"]]], ignore_index=True)
    df_features_combined.to_csv(ORIGINAL_TEST_FEATURES, index=False)
    print(f"[test] updated {ORIGINAL_TEST_FEATURES}: {len(df_orig_features)} -> {len(df_features_combined)} rows.")
    print(f"[test] class distribution in augmented test set (N = {len(df_features_combined)}):")
    print(df_features_combined["label"].value_counts())
    print(df_features_combined["label"].value_counts(normalize=True))


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Augment training and test sets with realistic deep-linked URLs.")
    parser.add_argument("--test", action="store_true", help="Augment held-out test set to 50,000 samples")
    parser.add_argument("--train", action="store_true", help="Augment training set with 50,000 deep links")
    parser.add_argument("--all", action="store_true", help="Augment both training and test sets")
    args = parser.parse_args()

    if args.all:
        augment_training_set()
        augment_test_set()
    elif args.test:
        augment_test_set()
    else:
        augment_training_set()


if __name__ == "__main__":
    main()
