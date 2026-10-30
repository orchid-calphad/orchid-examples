"""Tests of the repository hygiene rules and the quick-path table in tests/."""

from __future__ import annotations

import importlib
import sys
import tempfile
from pathlib import Path
from unittest import TestCase

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
hygiene = importlib.import_module("check_hygiene")
quick = importlib.import_module("check_quick_paths")


class TestLocalPathPattern(TestCase):
    """Tests for the absolute-local-path pattern of check_hygiene."""

    def test_flags_home_and_users_paths(self):
        """Paths under a user home (macOS, Linux, Windows) are matched."""
        for line in ("/Users/alice/x", "/home/bob/x", "C:\\\\Users\\carol\\x", "C:\\Users\\dave\\x", "~/projects/y"):
            assert hygiene.local_path_pattern.search(line), line

    def test_ignores_repository_relative_paths(self):
        """Repository-relative and data-release paths are not flagged."""
        for line in ("cached/gate.json", "inputs/evidence.json", "../common/orchid_data.py"):
            assert not hygiene.local_path_pattern.search(line), line


class TestFileProblems(TestCase):
    """Tests for check_hygiene.file_problems on a temporary tree."""

    def test_pdf_outside_figures_is_reported(self):
        """A PDF outside a figures/ folder is a hygiene problem."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "paper.pdf").write_bytes(b"%PDF")
            assert hygiene.file_problems("paper.pdf", root) == ["paper.pdf: PDF outside a figures/ folder"]

    def test_pdf_inside_figures_is_allowed(self):
        """A PDF inside a figures/ folder passes."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "figures").mkdir()
            (root / "figures" / "fig.pdf").write_bytes(b"%PDF")
            assert hygiene.file_problems("figures/fig.pdf", root) == []

    def test_local_path_in_text_is_reported_with_line_number(self):
        """A local absolute path in a text file is reported with its line number."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "run.md").write_text("ok\nsee /home/alice/data\n", encoding="utf-8")
            problems = hygiene.file_problems("run.md", root)
            assert len(problems) == 1
            assert problems[0].startswith("run.md:2: local absolute path")


class TestMetadataProblems(TestCase):
    """Tests for check_hygiene.metadata_problems."""

    def test_grants_and_funding_are_reported(self):
        """A grants key in .zenodo.json and a funding key in CITATION.cff are both reported."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / ".zenodo.json").write_text('{"grants": []}', encoding="utf-8")
            (root / "CITATION.cff").write_text("funding: x\n", encoding="utf-8")
            assert hygiene.metadata_problems(root) == [".zenodo.json has 'grants'", "CITATION.cff has a 'funding' key"]

    def test_clean_metadata_passes(self):
        """Metadata without grants or funding keys produces no problems."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / ".zenodo.json").write_text('{"title": "x"}', encoding="utf-8")
            (root / "CITATION.cff").write_text("title: x\n", encoding="utf-8")
            assert hygiene.metadata_problems(root) == []


class TestQuickPathTable(TestCase):
    """Tests for the example table read by check_quick_paths."""

    def test_every_example_has_outputs_and_no_private_keys(self):
        """Each configured example lists its outputs; keys starting with an underscore are dropped."""
        cfg = quick.load_config()
        assert cfg
        assert all(not k.startswith("_") for k in cfg)
        assert all(v["outputs"] for v in cfg.values())
