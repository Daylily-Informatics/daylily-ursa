from pathlib import Path


def _extract_css_block(css: str, selector: str) -> str:
    """Extract a simple `selector { ... }` block from a stylesheet string."""

    start = css.find(f"{selector} {{")
    assert start != -1, f"Could not find CSS block for selector: {selector}"

    end = css.find("}", start)
    assert end != -1, f"Could not find end of CSS block for selector: {selector}"

    return css[start : end + 1]


def test_main_css_table_cell_padding_reduced_and_tokenized():
    """Regression test: table cell padding should be reduced ~40% globally.

    We assert via CSS tokens (vars) so refactors to exact selectors remain flexible
    while preserving the intended padding size.
    """

    css = Path("static/css/main.css").read_text(encoding="utf-8")

    # Base tables (~40% reduction from 1.0rem / 1.5rem)
    assert "--spacing-table-cell-y: 0.6rem;" in css
    assert "--spacing-table-cell-x: 0.9rem;" in css
    assert "padding: var(--spacing-table-cell-y) var(--spacing-table-cell-x);" in css

    # Compact tables (worksets)
    assert "--spacing-table-cell-compact-y: 0.3rem;" in css
    assert "--spacing-table-cell-compact-x: 0.6rem;" in css
    assert (
        "padding: var(--spacing-table-cell-compact-y) var(--spacing-table-cell-compact-x);"
        in css
    )


def test_main_css_footer_is_not_forced_visible():
    """Regression test: footer should be part of normal flow, not fixed/sticky."""

    css = Path("static/css/main.css").read_text(encoding="utf-8")

    # Layout primitives that keep footer at bottom on short pages.
    assert "min-height: 100vh;" in _extract_css_block(css, "body")
    assert "display: flex;" in _extract_css_block(css, "body")
    assert "flex-direction: column;" in _extract_css_block(css, "body")
    assert "flex: 1;" in _extract_css_block(css, ".main")

    footer_block = _extract_css_block(css, ".footer")
    assert "margin-top: auto;" in footer_block
    assert "position: fixed" not in footer_block
    assert "position: sticky" not in footer_block


def test_base_template_footer_renders_after_main_content():
    """Regression test: footer should appear after main content in base layout."""

    html = Path("templates/base.html").read_text(encoding="utf-8")

    main_start = html.find('<main class="main">')
    assert main_start != -1
    main_end = html.find("</main>", main_start)
    assert main_end != -1

    footer_start = html.find('<footer class="footer">')
    assert footer_start != -1

    assert main_end < footer_start


def test_main_js_tooltip_position_is_clamped_to_viewport():
    """Regression test: tooltip placement should not overflow off-screen.

    The user-menu tooltip is in the upper-right, so without clamping this can
    render partially off the viewport.
    """

    js = Path("static/js/main.js").read_text(encoding="utf-8")

    assert "window.innerWidth" in js
    assert "window.innerHeight" in js
    assert "Math.max" in js
    assert "Math.min" in js
    assert "rect.bottom" in js

