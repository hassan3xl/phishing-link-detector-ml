"""Derives the suspicious-keyword list and high-risk-TLD set used by
`extract.py`, from `data/train.csv` ONLY.

This is the leakage safeguard for two of the Phase 3 features: rather than
hand-guessing a keyword list or TLD blocklist (which would be "my prior
knowledge," not evidence from data, and is also how a subtler form of
leakage sneaks in if that guess were ever informed by looking at test
examples), both lists are statistically derived from the training split and
written to JSON artifacts that `extract.py` loads at import time. Test data
is never read by this script.

Method for both: start from a candidate universe, keep only candidates with
(a) enough support in train to be statistically meaningful and (b) a
phishing-rate "lift" over the training base rate — i.e. URLs containing this
keyword / using this TLD are meaningfully more likely to be phishing than a
random train URL.

Run: python -m src.features.vocab
"""

import json
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

TRAIN_PATH = Path("data/train.csv")
ARTIFACT_DIR = Path("src/features/artifacts")

# Candidate universe of lure/security/brand-adjacent words commonly cited in
# phishing literature as appearing in login-harvesting or urgency-based
# phishing URLs. This is only a *candidate* list — membership in the final
# feature is decided by the data below, not by this list itself.
KEYWORD_CANDIDATES = [
    "login", "log-in", "signin", "sign-in", "verify", "verification",
    "secure", "security", "account", "update", "confirm", "confirmation",
    "banking", "password", "credential", "wallet", "billing", "invoice",
    "suspend", "suspended", "unlock", "limited", "urgent", "alert",
    "authenticate", "validate", "recover", "recovery", "reset", "activate",
    "click", "bonus", "gift", "prize", "winner", "claim", "support",
    "service", "id", "pay", "payment", "refund", "webscr", "customer",
]

MIN_KEYWORD_SUPPORT = 50
MIN_KEYWORD_LIFT = 1.3

MIN_TLD_SUPPORT = 30
MIN_TLD_LIFT = 1.5


def parsed_host(url: str) -> str:
    p = urlparse(url if "://" in url else "http://" + url)
    return (p.hostname or "").lower()


def derive_keywords(train: pd.DataFrame) -> dict:
    base_rate = (train["label"] == 0).mean()
    urls_lower = train["URL"].str.lower()
    rows = []
    for kw in KEYWORD_CANDIDATES:
        mask = urls_lower.str.contains(kw, regex=False)
        support = int(mask.sum())
        if support == 0:
            continue
        phishing_rate = float((train.loc[mask, "label"] == 0).mean())
        lift = phishing_rate / base_rate if base_rate > 0 else 0.0
        rows.append({
            "keyword": kw, "support": support,
            "phishing_rate": round(phishing_rate, 4), "lift": round(lift, 3),
        })
    stats = pd.DataFrame(rows).sort_values("lift", ascending=False)
    kept = stats[(stats["support"] >= MIN_KEYWORD_SUPPORT) & (stats["lift"] >= MIN_KEYWORD_LIFT)]
    print("\n[keywords] candidate stats (train-derived):")
    print(stats.to_string(index=False))
    print(f"\n[keywords] kept {len(kept)}/{len(stats)} candidates "
          f"(support>={MIN_KEYWORD_SUPPORT}, lift>={MIN_KEYWORD_LIFT}):")
    print(kept["keyword"].tolist())
    return {
        "base_rate": round(float(base_rate), 4),
        "min_support": MIN_KEYWORD_SUPPORT,
        "min_lift": MIN_KEYWORD_LIFT,
        "keywords": kept["keyword"].tolist(),
        "stats": stats.to_dict(orient="records"),
    }


def derive_tld_risk(train: pd.DataFrame) -> dict:
    base_rate = (train["label"] == 0).mean()
    hosts = train["URL"].apply(parsed_host)
    tlds = hosts.apply(lambda h: h.rsplit(".", 1)[-1] if "." in h else "")
    tmp = pd.DataFrame({"tld": tlds, "label": train["label"]})
    tmp = tmp[tmp["tld"] != ""]

    grp = tmp.groupby("tld")["label"].agg(support="count", phishing_rate=lambda s: (s == 0).mean())
    grp["phishing_rate"] = grp["phishing_rate"].round(4)
    grp["lift"] = (grp["phishing_rate"] / base_rate).round(3)
    grp = grp.sort_values("lift", ascending=False)

    eligible = grp[grp["support"] >= MIN_TLD_SUPPORT]
    kept = eligible[eligible["lift"] >= MIN_TLD_LIFT]

    print(f"\n[tld] {len(grp)} distinct TLDs in train; {len(eligible)} have "
          f"support>={MIN_TLD_SUPPORT}")
    print("\n[tld] top 15 eligible TLDs by lift:")
    print(eligible.head(15).to_string())
    print(f"\n[tld] flagged {len(kept)} high-risk TLDs (lift>={MIN_TLD_LIFT}):")
    print(kept.index.tolist())

    return {
        "base_rate": round(float(base_rate), 4),
        "min_support": MIN_TLD_SUPPORT,
        "min_lift": MIN_TLD_LIFT,
        "high_risk_tlds": kept.index.tolist(),
        "stats": eligible.reset_index().to_dict(orient="records"),
    }


def main():
    train = pd.read_csv(TRAIN_PATH)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    kw = derive_keywords(train)
    (ARTIFACT_DIR / "suspicious_keywords.json").write_text(json.dumps(kw, indent=2))

    tld = derive_tld_risk(train)
    (ARTIFACT_DIR / "tld_risk.json").write_text(json.dumps(tld, indent=2))

    print(f"\nArtifacts written to {ARTIFACT_DIR}/")


if __name__ == "__main__":
    main()
