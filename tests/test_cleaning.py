"""Tests for rag.cleaning (normalisation, header stripping, tokenising)."""

from __future__ import annotations

from rag.cleaning import clean_text, content_tokens, normalise_text, strip_repeated_lines, tokens


def test_normalise_applies_nfk_c():
    assert normalise_text("cafe\u0301") == "caf\u00e9"   # e + combining acute -> é
    assert normalise_text("\uff21") == "A"               # fullwidth A -> A


def test_normalise_collapses_whitespace_keeps_paragraphs():
    assert normalise_text("a   b\t\tc") == "a b c"
    assert normalise_text("p1\n\n\n\np2") == "p1\n\np2"
    assert normalise_text("  trailing spaces \n\n") == "trailing spaces"
    assert normalise_text("para one\n\npara two").count("\n\n") == 1


def test_strip_repeated_headers():
    blocks = [f"para {i} body sentence about refunds.\nCONFIDENTIAL FOOTER"
              for i in range(4)]
    text = "\n\n".join(blocks)
    cleaned = strip_repeated_lines(text)
    assert "CONFIDENTIAL FOOTER" not in cleaned
    assert "para 2 body sentence about refunds." in cleaned


def test_repeated_first_lines_stripped():
    text = "\n\n".join(["HDR LINE\nbody 0", "HDR LINE\nbody 10",
                        "HDR LINE\nbody last"])
    cleaned = strip_repeated_lines(text)
    assert "HDR LINE" not in cleaned
    assert "body 10" in cleaned and "body last" in cleaned


def test_no_repeats_means_no_change():
    text = "first\n\nA paragraph here.\n\nsecond header\n\nB paragraph here."
    assert strip_repeated_lines(text) == text


def test_fewer_than_min_blocks_means_no_change():
    text = "H\n\nA.\n\nH\n\nB."
    assert strip_repeated_lines(text) == text


def test_tokenizer():
    assert tokens("State-of-the-art INR 2,500!") == \
        ["state-of-the-art", "inr", "2", "500"]
    assert tokens("RefundPolicy") == ["refundpolicy"]
    assert content_tokens("The refund and the policy") == ["refund", "policy"]


def test_clean_text_pipeline_removes_furniture():
    raw = "H\n\nH\n\nH\n\nReal policy content here."
    assert clean_text(raw) == "Real policy content here."
