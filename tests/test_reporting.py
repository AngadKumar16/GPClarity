"""
Tests for one-call diagnostic reporting.

Report generation degrades gracefully, so most behaviour is checked with GPy
models; export/format logic is checked with a hand-built report.
"""

import json

import numpy as np
import pytest

import gpclarity
from gpclarity.exceptions import ReportError
from gpclarity.reporting import DiagnosticReport, generate_report


class TestExportFormats:
    def _report(self):
        return DiagnosticReport(
            title="Test Report",
            created="2026-01-01T00:00:00",
            sections={
                "health": {"ok": True, "data": {"healthy": True, "issues": []}},
                "broken": {"ok": False, "error": "boom"},
            },
            summary=["Model health: OK"],
        )

    def test_to_markdown(self):
        md = self._report().to_markdown()
        assert "# Test Report" in md
        assert "## Summary" in md
        assert "Not available: boom" in md

    def test_to_html_escapes(self):
        r = self._report()
        r.sections["x"] = {"ok": True, "data": {"k": "<script>"}}
        html = r.to_html()
        assert "<!doctype html>" in html
        assert "&lt;script&gt;" in html
        assert "<script>" not in html.split("<style>")[1]

    def test_to_json_roundtrip(self):
        r = self._report()
        parsed = json.loads(r.to_json())
        assert parsed["title"] == "Test Report"
        assert parsed["sections"]["broken"]["ok"] is False

    def test_json_handles_numpy(self):
        r = DiagnosticReport(
            title="np", created="now",
            sections={"s": {"ok": True, "data": {"arr": np.arange(3),
                                                 "val": np.float64(1.5)}}},
        )
        parsed = json.loads(r.to_json())
        assert parsed["sections"]["s"]["data"]["arr"] == [0, 1, 2]

    def test_save_dispatch(self, tmp_path):
        r = self._report()
        for ext in ("md", "html", "json"):
            p = tmp_path / f"report.{ext}"
            r.save(str(p))
            assert p.exists() and p.read_text()

    def test_save_bad_ext(self, tmp_path):
        with pytest.raises(ReportError):
            self._report().save(str(tmp_path / "report.txt"))


class TestGenerate:
    def test_none_model(self):
        with pytest.raises(ReportError):
            generate_report(None)

    def test_full_report(self, simple_gp):
        X = simple_gp.X
        y = simple_gp.Y.ravel()
        X_test = np.linspace(-2, 12, 60).reshape(-1, 1)
        report = gpclarity.generate_report(
            simple_gp, X=X, y=y, X_test=X_test, title="Full"
        )
        assert report.title == "Full"
        # health + kernel + complexity + uncertainty + influence
        assert set(report.sections) >= {"health", "kernel", "complexity"}
        assert report.sections["health"]["ok"]
        assert any("health" in s.lower() for s in report.summary)
        # renders without error
        assert "# Full" in report.to_markdown()
        assert "<html>" in report.to_html()

    def test_include_whitelist(self, simple_gp):
        report = gpclarity.generate_report(
            simple_gp, X=simple_gp.X, include=["health", "kernel"]
        )
        assert set(report.sections) == {"health", "kernel"}

    def test_partial_without_data(self, simple_gp):
        # no X -> complexity/uncertainty/influence skipped, health/kernel present
        report = gpclarity.generate_report(simple_gp)
        assert "health" in report.sections
        assert "uncertainty" not in report.sections
