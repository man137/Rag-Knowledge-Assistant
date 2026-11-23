"""Tests for rag.loaders (front matter, md/txt/pdf, skip-on-error)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from rag.loaders import load_directory, load_file, parse_front_matter

DOCS_DIR = Path(__file__).resolve().parents[1] / "data" / "docs"


def test_front_matter_parsing():
    text = ("---\ntitle: Test Policy\nsource: tp.md\n"
            "allowed_roles: [staff, manager]\n---\n\nBody text.")
    meta, body = parse_front_matter(text)
    assert meta["title"] == "Test Policy"
    assert meta["allowed_roles"] == ["staff", "manager"]
    assert body == "Body text."


def test_front_matter_comma_separated_roles():
    meta, _ = parse_front_matter("---\nallowed_roles: all, hr\n---\nx")
    assert meta["allowed_roles"] == ["all", "hr"]


def test_unterminated_front_matter_is_plain_text():
    meta, body = parse_front_matter("---\ntitle: X\nbody without close")
    assert meta == {}
    assert body.startswith("---")


def test_missing_allowed_roles_defaults_to_all(tmp_path):
    p = tmp_path / "doc.md"
    p.write_text("---\ntitle: Open Note\n---\n\nhello world", encoding="utf-8")
    (doc,) = load_file(p)
    assert doc.allowed_roles == ["all"]
    assert doc.title == "Open Note"
    assert doc.text == "hello world"
    assert doc.page == 1


def test_txt_loader(tmp_path):
    p = tmp_path / "note.txt"
    p.write_text("plain text content", encoding="utf-8")
    (doc,) = load_file(p)
    assert doc.text == "plain text content"
    assert doc.metadata["allowed_roles"] == ["all"]


def test_unsupported_suffix_skipped(tmp_path):
    p = tmp_path / "blob.bin"
    p.write_text("x", encoding="utf-8")
    assert load_file(p) == []


def test_unreadable_file_skipped(tmp_path):
    p = tmp_path / "bad.md"
    p.write_bytes(b"\xff\xfe\x00garbage")  # not valid UTF-8
    assert load_file(p) == []


def test_pdf_skipped_with_warning_when_pypdf_missing(tmp_path, monkeypatch, caplog):
    # Setting sys.modules["pypdf"] to None makes `import pypdf` raise ImportError.
    monkeypatch.setitem(sys.modules, "pypdf", None)
    p = tmp_path / "x.pdf"
    p.write_bytes(b"%PDF-1.4 junk")
    with caplog.at_level("WARNING"):
        assert load_file(p) == []
    assert any("pypdf" in record.getMessage() for record in caplog.records)


def _build_pdf(text: str) -> bytes:
    """Hand-build a valid single-page PDF (correct xref offsets) so the test
    needs no external fixture file. Helvetica text only; no parentheses in
    the payload text."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("ascii")
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n"
        + stream + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref_at = len(out)
    out += b"xref\n0 6\n0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n"
            + str(xref_at).encode() + b"\n%%EOF\n")
    return bytes(out)


def test_pdf_page_extraction(tmp_path):
    pytest.importorskip("pypdf")
    p = tmp_path / "sample.pdf"
    p.write_bytes(_build_pdf("Synthetic PDF test page. Refunds need 14 business days."))
    (doc,) = load_file(p)
    assert doc.page == 1
    assert "Refunds need 14 business days." in doc.text


def test_load_directory_loads_the_six_synthetic_docs():
    docs = load_directory(DOCS_DIR)
    assert len(docs) == 6
    by_name = {Path(d.source).name: d for d in docs}
    assert by_name["travel_expense_policy.md"].allowed_roles == ["all"]
    assert by_name["event_registration_faq.md"].allowed_roles == ["all"]
    assert by_name["venue_booking_sop.md"].allowed_roles == ["staff", "manager"]
    assert by_name["visa_travel_advisory_faq.md"].allowed_roles == ["all"]
    assert by_name["support_escalation_sop.md"].allowed_roles == ["staff", "manager"]
    assert by_name["hr_compensation_leave_policy.md"].allowed_roles == ["hr"]


def test_load_directory_missing_dir_is_empty(tmp_path):
    assert load_directory(tmp_path / "nope") == []
