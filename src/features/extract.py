"""Phase 3: Feature engineering.

`extract_features(url)` turns a raw URL string into a dict of numeric/boolean
features. This exact function is reused unchanged by the FastAPI service in
Phase 7 (train/serve skew safeguard - see README) so it deliberately has no
dependency on anything outside the standard library plus the two small JSON
artifacts derived from `data/train.csv` in `vocab.py`.

Every feature below carries a one-sentence rationale in its section comment;
these are compiled verbatim into the README's feature data dictionary.
"""

import json
import math
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

ARTIFACT_DIR = Path(__file__).parent / "artifacts"

_IP_PATTERN = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$|^0x[0-9a-fA-F]+$")
_STRUCTURAL_CHARS = set(":/?#[]@")  # standard URL delimiters, not "special" for our purposes


def _load_json(name: str) -> dict:
    path = ARTIFACT_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run `python -m src.features.vocab` first "
            "to derive it from data/train.csv."
        )
    return json.loads(path.read_text())


_KEYWORDS = _load_json("suspicious_keywords.json")["keywords"]
_HIGH_RISK_TLDS = set(_load_json("tld_risk.json")["high_risk_tlds"])


def _parsed(url: str):
    """Normalizes a URL missing a scheme (e.g. 'example.com/x') by assuming
    http://, so downstream parsing is consistent whether or not the caller
    included a scheme - matters for the API in Phase 7, where user input may
    omit it."""
    return urlparse(url if "://" in url else "http://" + url)


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def extract_features(url: str) -> dict:
    """Extracts the full Phase 3 feature set from a single raw URL string."""
    url = (url or "").strip()
    parsed = _parsed(url)
    host = (parsed.hostname or "").lower()
    # Second-to-last dot-separated label, e.g. "google" from "www.google.com"
    # or from "google.com". This is a simplification that doesn't handle
    # multi-part TLDs (e.g. "co.uk" would make "google" the wrong pick for
    # "www.google.co.uk", which yields "google" here too since it's still
    # the second-to-last label - not "google.co"). Acceptable for this
    # project's scope; a proper public-suffix-list lookup is future work.
    labels = host.split(".") if host else []
    domain_no_tld = labels[-2] if len(labels) >= 2 else (labels[0] if labels else "")

    features = {}

    # url_length: total character count.
    # Rationale: phishing URLs often embed long random tokens or redirect
    # paths to evade filters and to bury the real destination (EDA: mean 46
    # chars phishing vs. 27 legitimate in this dataset's train split).
    features["url_length"] = len(url)

    # domain_length: character count of the hostname only.
    # Rationale: isolates length signal to the domain itself, since attacker-
    # registered domains are sometimes unusually long (subdomain stuffing)
    # independent of path length.
    features["domain_length"] = len(host)

    # num_subdomains: dot-count-based subdomain depth heuristic.
    # Rationale: attackers pile on subdomains (e.g. "paypal.com.verify.xyz")
    # to make a fake domain visually resemble a trusted brand's domain.
    features["num_subdomains"] = max(host.count(".") - 1, 0) if host else 0

    # is_ip_address: hostname is a raw IPv4/hex-encoded IP rather than a name.
    # Rationale: legitimate consumer sites are addressed by registered domain
    # names; a bare IP as the host is a classic way to sidestep domain-based
    # blocklists and WHOIS-based reputation checks.
    features["is_ip_address"] = int(bool(_IP_PATTERN.match(host)))

    # is_https: URL scheme is https.
    # Rationale: attackers have historically been less likely to pay for/set
    # up valid HTTPS certificates than established sites (though this gap is
    # narrowing industry-wide; still a strong signal in this dataset - see
    # EDA finding #1, with the caveat about dataset-construction artifacts).
    features["is_https"] = int(parsed.scheme == "https")

    # count_dots: total '.' characters in the full URL.
    # Rationale: correlates with both subdomain depth and path complexity;
    # a cheap proxy for "how many name segments is this URL threading
    # through."
    features["count_dots"] = url.count(".")

    # count_hyphens_domain: '-' characters in the hostname.
    # Rationale: brand-impersonation domains often insert hyphens between
    # words to squat near a trademark (e.g. "paypal-secure-login.com").
    features["count_hyphens_domain"] = host.count("-")

    # count_at: '@' characters anywhere in the URL.
    # Rationale: browsers ignore everything before an '@' as userinfo, so
    # attackers exploit it to disguise the real destination host behind what
    # looks like a trusted prefix.
    features["count_at"] = url.count("@")

    # count_digits_domain / digit_ratio_domain: digit count and ratio within
    # the hostname.
    # Rationale: algorithmically generated or freshly-registered attacker
    # domains often contain digits atypical of real brand names.
    digits_in_domain = sum(c.isdigit() for c in host)
    features["count_digits_domain"] = digits_in_domain
    features["digit_ratio_domain"] = digits_in_domain / len(host) if host else 0.0

    # special_char_ratio_url: ratio of non-alphanumeric, non-structural
    # characters (excludes ':','/','?','#','[',']','@') to URL length.
    # Rationale: phishing URLs lean on heavier symbol usage (%, _, =, &) from
    # query-string obfuscation and encoded redirect payloads.
    special = sum(
        1 for c in url if not c.isalnum() and c not in _STRUCTURAL_CHARS and c != "."
    )
    features["special_char_ratio_url"] = special / len(url) if url else 0.0

    # has_suspicious_keyword / num_suspicious_keywords: presence/count of
    # lure keywords (login, verify, secure, account, ...) derived from
    # data/train.csv only (see vocab.py) - not hand-picked, and never
    # touching test data, per the leakage safeguard.
    url_lower = url.lower()
    hits = [kw for kw in _KEYWORDS if kw in url_lower]
    features["has_suspicious_keyword"] = int(len(hits) > 0)
    features["num_suspicious_keywords"] = len(hits)

    # url_entropy: Shannon entropy (bits/char) of the full URL string.
    # Rationale: algorithmically generated phishing domains/paths look more
    # "random" than human-chosen brand names and dictionary words, which
    # shows up as higher character-level entropy.
    features["url_entropy"] = round(_shannon_entropy(url), 4)

    # domain_entropy: Shannon entropy of the second-level domain label alone
    # (e.g. "google" from "www.google.com" - see the simplification note
    # above).
    # Rationale: isolates the randomness signal to the domain itself (where
    # DGA-like phishing domains are most likely to show it) without it being
    # diluted by an arbitrary, often-legitimate-looking path or subdomain.
    features["domain_entropy"] = round(_shannon_entropy(domain_no_tld), 4)

    # tld_risk_flag: hostname's TLD is in the high-risk set derived from
    # train.csv (TLDs with a train-set phishing rate ≥1.5x the base rate,
    # e.g. .xyz, .top, .icu, .gq - see vocab.py for the exact method/list).
    # Rationale: certain cheap, loosely-vetted TLDs are disproportionately
    # abused by phishers because registration is cheap and fast; note this
    # is also this dataset's least brand-safe feature - several benign TLDs
    # popular with startups (.io, .co, .me) ended up in the flagged set too,
    # a limitation discussed in the README.
    tld = host.rsplit(".", 1)[-1] if "." in host else ""
    features["tld_risk_flag"] = int(tld in _HIGH_RISK_TLDS)

    # count_slashes_path: '/' characters after the domain (path depth).
    # Rationale: unusually deep or padded path structure can be used to bury
    # a malicious destination or mimic a legitimate multi-page site's URL
    # shape.
    path = parsed.path or ""
    features["count_slashes_path"] = path.count("/")

    # NOTE: an earlier version of this feature set also included `has_port`
    # (explicit non-default port in the URL). It was dropped after the
    # variance/redundancy check in build_matrix.py: only 0.011% of training
    # rows had a nonzero value and its correlation with label was ~0 (-0.01)
    # - on this dataset it was pure noise with overfitting risk from ~20
    # supporting rows, not a useful signal, so it's excluded from the final
    # feature set. See the README's Feature Engineering section for the full
    # variance/correlation check writeup.

    return features


FEATURE_NAMES = [
    "url_length", "domain_length", "num_subdomains", "is_ip_address",
    "is_https", "count_dots", "count_hyphens_domain", "count_at",
    "count_digits_domain", "digit_ratio_domain", "special_char_ratio_url",
    "has_suspicious_keyword", "num_suspicious_keywords", "url_entropy",
    "domain_entropy", "tld_risk_flag", "count_slashes_path",
]
