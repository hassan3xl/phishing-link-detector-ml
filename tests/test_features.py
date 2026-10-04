"""Tests for src.features.extract.extract_features - hand-crafted URLs with
known expected feature values, per Phase 8 of the build guide."""

from src.features.extract import FEATURE_NAMES, extract_features


def test_returns_all_expected_features():
    f = extract_features("https://www.google.com")
    assert set(f.keys()) == set(FEATURE_NAMES)


def test_ip_address_domain_is_flagged():
    f = extract_features("http://192.168.1.1/login")
    assert f["is_ip_address"] == 1


def test_domain_name_is_not_flagged_as_ip():
    f = extract_features("https://www.google.com")
    assert f["is_ip_address"] == 0


def test_https_scheme_detected():
    assert extract_features("https://example.com")["is_https"] == 1
    assert extract_features("http://example.com")["is_https"] == 0


def test_at_symbol_counted():
    f = extract_features("http://trusted-brand.com@evil.com/login")
    assert f["count_at"] == 1
    assert extract_features("https://example.com")["count_at"] == 0


def test_url_length_matches_string_length():
    url = "https://www.example.com/some/path"
    assert extract_features(url)["url_length"] == len(url)


def test_suspicious_keyword_detected_in_login_url():
    f = extract_features("http://example.com/login")
    assert f["has_suspicious_keyword"] == 1
    assert f["num_suspicious_keywords"] >= 1


def test_no_suspicious_keyword_in_plain_url():
    f = extract_features("https://www.wikipedia.org")
    assert f["has_suspicious_keyword"] == 0
    assert f["num_suspicious_keywords"] == 0


def test_high_risk_tld_flagged():
    # .xyz is in the train-derived high-risk TLD set (src/features/artifacts/tld_risk.json)
    f = extract_features("http://totally-fake-brand.xyz")
    assert f["tld_risk_flag"] == 1


def test_common_tld_not_flagged_as_high_risk():
    f = extract_features("https://www.google.com")
    assert f["tld_risk_flag"] == 0


def test_subdomain_count():
    assert extract_features("https://www.google.com")["num_subdomains"] == 1
    assert extract_features("https://a.b.c.google.com")["num_subdomains"] == 3
    assert extract_features("https://google.com")["num_subdomains"] == 0


def test_entropy_is_nonnegative():
    f = extract_features("http://xk29fjq02mz.tk/verify")
    assert f["url_entropy"] >= 0
    assert f["domain_entropy"] >= 0


def test_entropy_zero_for_empty_or_single_char_repeats():
    # A single-character-repeated domain label has zero Shannon entropy.
    f = extract_features("http://aaa.com")
    assert f["domain_entropy"] == 0.0


def test_handles_url_missing_scheme():
    # Should not raise, and should behave like an implicit http:// prefix.
    f = extract_features("www.example.com/path")
    assert f["url_length"] == len("www.example.com/path")


def test_handles_empty_string_without_raising():
    f = extract_features("")
    assert f["url_length"] == 0
    assert f["is_ip_address"] == 0


def test_count_slashes_path():
    assert extract_features("https://example.com")["count_slashes_path"] == 0
    assert extract_features("https://example.com/a/b/c")["count_slashes_path"] == 3
