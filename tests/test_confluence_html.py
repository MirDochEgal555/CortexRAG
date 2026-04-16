"""Tests for Confluence HTML export preprocessing."""

from __future__ import annotations

from pathlib import Path
import sys
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cortex_rag.ingestion.confluence_html import preprocess_confluence_archive


def test_preprocess_confluence_archive_creates_markdown_pages(tmp_path: Path) -> None:
    zip_path = tmp_path / "ASA_2026-04-16.zip"
    output_dir = tmp_path / "processed"

    index_html = """<!DOCTYPE html>
<html>
  <head><title>ASA (AI Sales Agent)</title></head>
  <body>
    <div id="page">
      <div id="content">
        <div id="main-content" class="pageSection">
          <table class="confluenceTable">
            <tr><th>Key</th><td>ASA</td></tr>
            <tr><th>Name</th><td>AI Sales Agent</td></tr>
          </table>
        </div>
        <div class="pageSection">
          <h2>Available Pages:</h2>
          <ul><li><a href="Overview_3178688.html">Overview</a></li></ul>
        </div>
      </div>
    </div>
  </body>
</html>
"""
    page_html = """<!DOCTYPE html>
<html>
  <head><title>AI Sales Agent : Overview</title></head>
  <body>
    <ol id="breadcrumbs">
      <li><a href="index.html">AI Sales Agent</a></li>
      <li><a href="AI-Sales-Agent-Home_3178669.html">AI Sales Agent Home</a></li>
    </ol>
    <span id="title-text">AI Sales Agent : Overview</span>
    <div class="page-metadata">Created by <span class="author">Robin Keim</span> on Mar 28, 2026</div>
    <div id="main-content">
      <p># Product Overview</p>
      <p>- First item</p>
      <p><a href="AI-Sales-Agent-Home_3178669.html">Home</a></p>
    </div>
  </body>
</html>
"""
    home_html = """<!DOCTYPE html>
<html>
  <head><title>AI Sales Agent : Home</title></head>
  <body>
    <ol id="breadcrumbs">
      <li><a href="index.html">AI Sales Agent</a></li>
    </ol>
    <span id="title-text">AI Sales Agent : Home</span>
    <div id="main-content"><h1>Home</h1></div>
  </body>
</html>
"""

    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("ASA/index.html", index_html)
        archive.writestr("ASA/Overview_3178688.html", page_html)
        archive.writestr("ASA/AI-Sales-Agent-Home_3178669.html", home_html)

    output_paths = preprocess_confluence_archive(zip_path, output_dir)

    assert len(output_paths) == 3

    overview_path = output_dir / "ASA" / "overview-3178688.md"
    assert overview_path.exists()
    overview_text = overview_path.read_text(encoding="utf-8")
    assert 'space_name: "AI Sales Agent"' in overview_text
    assert 'created_on: "2026-03-28"' in overview_text
    assert "# Product Overview" in overview_text
    assert "- First item" in overview_text
    assert "[Home](home-3178669.md)" in overview_text

    index_path = output_dir / "ASA" / "space-index.md"
    index_text = index_path.read_text(encoding="utf-8")
    assert "- Key: ASA" in index_text
    assert "[Overview](overview-3178688.md)" in index_text
