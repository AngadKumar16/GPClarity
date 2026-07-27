"""
One-call diagnostic reports for Gaussian Process models.

:func:`generate_report` runs the full GPClarity battery — health check, kernel
summary, complexity score, uncertainty diagnostics, and data influence — against
a single model and packages the results into a :class:`DiagnosticReport` that can
be printed, exported to Markdown or self-contained HTML, or dumped as JSON.

Each section is computed defensively: if one analysis fails (or its optional
dependency is missing) the report records the error for that section and
continues, so you always get a complete document.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from gpclarity.exceptions import ReportError

logger = logging.getLogger(__name__)


def _safe(section: str, fn, *args, **kwargs) -> Dict[str, Any]:
    """Run an analysis callable, capturing failures as a section error.

    Args:
        section: Section name (for logging).
        fn: Callable to invoke.
        *args: Positional arguments for ``fn``.
        **kwargs: Keyword arguments for ``fn``.

    Returns:
        ``{"ok": True, "data": result}`` on success, otherwise
        ``{"ok": False, "error": message}``.
    """
    try:
        return {"ok": True, "data": fn(*args, **kwargs)}
    except Exception as e:  # noqa: BLE001 - report, don't crash
        logger.warning("report section %r failed: %s", section, e)
        return {"ok": False, "error": str(e)}


@dataclass
class DiagnosticReport:
    """A structured, multi-section diagnostic report for one model.

    Attributes:
        title: Report title.
        created: ISO-8601 timestamp of generation.
        sections: Mapping of section name → result payload. Each payload has an
            ``ok`` flag plus either ``data`` or ``error``.
        summary: Short headline verdicts distilled from the sections.
    """

    title: str
    created: str
    sections: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    summary: List[str] = field(default_factory=list)

    # -- serialisation -----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Return the whole report as a JSON-ready dictionary."""
        return {
            "title": self.title,
            "created": self.created,
            "summary": self.summary,
            "sections": self.sections,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialise the report to a JSON string.

        Args:
            indent: Indentation passed to :func:`json.dumps`.

        Returns:
            JSON text. NumPy scalars/arrays are coerced to native types.
        """

        def _default(o: Any) -> Any:
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            return str(o)

        return json.dumps(self.to_dict(), indent=indent, default=_default)

    def to_markdown(self) -> str:
        """Render the report as GitHub-flavoured Markdown."""
        out: List[str] = [f"# {self.title}", "", f"*Generated: {self.created}*", ""]
        if self.summary:
            out.append("## Summary")
            out += [f"- {s}" for s in self.summary]
            out.append("")
        for name, payload in self.sections.items():
            out.append(f"## {name.replace('_', ' ').title()}")
            if not payload.get("ok"):
                out.append(f"> Not available: {payload.get('error', 'unknown error')}")
                out.append("")
                continue
            out += _render_lines(payload["data"])
            out.append("")
        return "\n".join(out)

    def to_html(self) -> str:
        """Render the report as a minimal, self-contained HTML document."""
        css = (
            "body{font-family:system-ui,Arial,sans-serif;max-width:840px;"
            "margin:2rem auto;padding:0 1rem;color:#1a1a1a;line-height:1.5}"
            "h1{border-bottom:3px solid #2980B9}h2{margin-top:1.8rem;color:#2980B9}"
            "code,pre{background:#f4f6f8;border-radius:4px}pre{padding:.75rem;"
            "overflow:auto}.err{color:#b00020}ul{margin:.3rem 0}"
        )
        parts: List[str] = [
            "<!doctype html><html><head><meta charset='utf-8'>",
            f"<title>{_esc(self.title)}</title><style>{css}</style></head><body>",
            f"<h1>{_esc(self.title)}</h1><p><em>Generated: {_esc(self.created)}</em></p>",
        ]
        if self.summary:
            parts.append("<h2>Summary</h2><ul>")
            parts += [f"<li>{_esc(s)}</li>" for s in self.summary]
            parts.append("</ul>")
        for name, payload in self.sections.items():
            parts.append(f"<h2>{_esc(name.replace('_', ' ').title())}</h2>")
            if not payload.get("ok"):
                parts.append(
                    f"<p class='err'>Not available: {_esc(payload.get('error', ''))}</p>"
                )
                continue
            parts.append("<pre>" + _esc("\n".join(_render_lines(payload["data"]))) + "</pre>")
        parts.append("</body></html>")
        return "".join(parts)

    def save(self, path: str) -> str:
        """Write the report to disk, picking the format from the extension.

        Args:
            path: Destination path ending in ``.md``, ``.markdown``, ``.html``,
                ``.htm``, or ``.json``.

        Returns:
            The path written.

        Raises:
            ReportError: If the extension is unrecognised.
        """
        lower = path.lower()
        if lower.endswith((".md", ".markdown")):
            text = self.to_markdown()
        elif lower.endswith((".html", ".htm")):
            text = self.to_html()
        elif lower.endswith(".json"):
            text = self.to_json()
        else:
            raise ReportError(
                f"unrecognised report extension: {path!r} "
                "(use .md, .html, or .json)"
            )
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return path

    def __str__(self) -> str:
        """Return the Markdown rendering."""
        return self.to_markdown()


def _esc(text: Any) -> str:
    """HTML-escape a value's string form."""
    s = str(text)
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _render_lines(data: Any, depth: int = 0) -> List[str]:
    """Render an arbitrary payload as indented Markdown bullet lines.

    Args:
        data: A dict, list, or scalar produced by an analysis.
        depth: Current nesting depth (controls indentation).

    Returns:
        A list of Markdown lines.
    """
    pad = "  " * depth
    lines: List[str] = []
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{pad}- **{k}:**")
                lines += _render_lines(v, depth + 1)
            else:
                lines.append(f"{pad}- **{k}:** {_fmt(v)}")
    elif isinstance(data, (list, tuple)):
        for item in data:
            if isinstance(item, (dict, list)):
                lines += _render_lines(item, depth + 1)
            else:
                lines.append(f"{pad}- {_fmt(item)}")
    else:
        lines.append(f"{pad}{_fmt(data)}")
    return lines


def _fmt(v: Any) -> str:
    """Format a scalar for display, rounding floats to 4 significant figures."""
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.4g}"
    if isinstance(v, np.ndarray):
        return np.array2string(v, precision=4, threshold=8)
    return str(v)


def generate_report(
    model: Any,
    X: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    X_test: Optional[np.ndarray] = None,
    title: str = "GPClarity Diagnostic Report",
    include: Optional[List[str]] = None,
) -> DiagnosticReport:
    """Run all diagnostics on a model and assemble a :class:`DiagnosticReport`.

    Sections (each optional and computed defensively): ``health``, ``kernel``,
    ``complexity``, ``uncertainty`` (needs ``X`` and ``X_test``), and
    ``influence`` (needs ``X``).

    Args:
        model: A trained GP model.
        X: Training inputs; enables complexity, uncertainty, and influence.
        y: Training targets; improves the influence section when provided.
        X_test: Test grid for the uncertainty section.
        title: Report title.
        include: Optional whitelist of section names to run. Defaults to all
            applicable sections.

    Returns:
        A populated :class:`DiagnosticReport`.

    Raises:
        ReportError: If ``model`` is ``None``.

    Example:
        >>> report = gpclarity.generate_report(model, X=X, y=y, X_test=Xt)
        >>> report.save("diagnostics.html")
    """
    if model is None:
        raise ReportError("model cannot be None")

    # Imported lazily so a missing optional dep only disables its own section.
    from gpclarity import (
        DataInfluenceMap,
        UncertaintyProfiler,
        check_model_health,
        compute_complexity_score,
        summarize_kernel,
    )

    all_sections = ["health", "kernel", "complexity", "uncertainty", "influence"]
    wanted = set(include) if include else set(all_sections)

    sections: Dict[str, Dict[str, Any]] = {}

    if "health" in wanted:
        sections["health"] = _safe("health", check_model_health, model)

    if "kernel" in wanted:
        sections["kernel"] = _safe("kernel", summarize_kernel, model)

    if "complexity" in wanted and X is not None:
        sections["complexity"] = _safe(
            "complexity", compute_complexity_score, model, X
        )

    if "uncertainty" in wanted and X is not None and X_test is not None:
        def _uncertainty() -> Dict[str, Any]:
            prof = UncertaintyProfiler(model, X_train=X)
            return prof.compute_diagnostics(X_test)

        sections["uncertainty"] = _safe("uncertainty", _uncertainty)

    if "influence" in wanted and X is not None:
        def _influence() -> Dict[str, Any]:
            infl = DataInfluenceMap(model)
            if y is not None:
                return infl.get_influence_report(X, y)
            return infl.get_influence_report(X)

        sections["influence"] = _safe("influence", _influence)

    report = DiagnosticReport(
        title=title,
        created=_dt.datetime.now().isoformat(timespec="seconds"),
        sections=sections,
    )
    report.summary = _build_summary(sections)
    return report


def _build_summary(sections: Dict[str, Dict[str, Any]]) -> List[str]:
    """Distil a few headline verdicts from the computed sections.

    Args:
        sections: The section payloads assembled by :func:`generate_report`.

    Returns:
        A short list of human-readable summary bullets.
    """
    out: List[str] = []

    health = sections.get("health", {})
    if health.get("ok"):
        d = health["data"]
        out.append(
            "Model health: OK"
            if d.get("healthy")
            else f"Model health: {len(d.get('issues', []))} issue(s) found"
        )

    comp = sections.get("complexity", {})
    if comp.get("ok"):
        d = comp["data"]
        cat = d.get("category") or d.get("interpretation")
        if cat is not None:
            out.append(f"Complexity: {cat}")

    unc = sections.get("uncertainty", {})
    if unc.get("ok"):
        d = unc["data"]
        n_ex = d.get("n_extrapolation_points")
        if n_ex is not None:
            out.append(f"Extrapolation points on test grid: {n_ex}")

    return out
