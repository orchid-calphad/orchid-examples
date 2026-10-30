"""Tests of the shared helpers in common/ (release-file lookup, hashing, deterministic figure output)."""

from __future__ import annotations

import hashlib
import importlib
import os
import sys
import tempfile
from pathlib import Path
from unittest import TestCase, mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "common"))
orchid_data = importlib.import_module("orchid_data")
figstyle = importlib.import_module("figstyle")
plt = importlib.import_module("matplotlib.pyplot")


class TestSha256(TestCase):
    """Tests for orchid_data.sha256."""

    def test_matches_hashlib_on_file_bytes(self):
        """The digest of a file equals hashlib's sha256 of its bytes."""
        payload = b"orchid" * 300000  # larger than one hashing chunk
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.bin"
            p.write_bytes(payload)
            assert orchid_data.sha256(p) == hashlib.sha256(payload).hexdigest()

    def test_empty_file_has_the_empty_digest(self):
        """An empty file hashes to the sha256 of the empty byte string."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "empty"
            p.write_bytes(b"")
            assert orchid_data.sha256(p) == hashlib.sha256(b"").hexdigest()


class TestLocalRelease(TestCase):
    """Tests for release-file lookup with ORCHID_DATA pointing at a local checkout."""

    def test_path_returns_the_local_file(self):
        """A release file under ORCHID_DATA resolves to that file."""
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {"ORCHID_DATA": d}):
            (Path(d) / "espei").mkdir()
            (Path(d) / "espei" / "index.csv").write_text("a\n")
            assert orchid_data.path("espei/index.csv", "0" * 40) == Path(d) / "espei" / "index.csv"

    def test_path_raises_for_a_missing_file(self):
        """A release file absent from the local checkout raises FileNotFoundError instead of downloading."""
        with (
            tempfile.TemporaryDirectory() as d,
            mock.patch.dict(os.environ, {"ORCHID_DATA": d}),
            self.assertRaises(FileNotFoundError),
        ):
            orchid_data.path("espei/missing.json", "0" * 40)

    def test_listdir_filters_by_suffix_and_sorts(self):
        """List the sorted matching files of a local folder, relative to the release root."""
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {"ORCHID_DATA": d}):
            folder = Path(d) / "espei" / "x"
            folder.mkdir(parents=True)
            for name in ("b.json", "a.json", "c.csv"):
                (folder / name).write_text("{}")
            assert orchid_data.listdir("espei/x", "0" * 40) == ["espei/x/a.json", "espei/x/b.json"]

    def test_local_root_is_none_when_unset(self):
        """Without ORCHID_DATA the local checkout is not used."""
        with mock.patch.dict(os.environ, {}):
            os.environ.pop("ORCHID_DATA", None)
            assert orchid_data.local_root() is None


class TestFigstyleSave(TestCase):
    """Tests for figstyle.save, which must write byte-identical files on repeated calls."""

    def test_repeated_saves_are_byte_identical(self):
        """Saving the same figure twice gives identical PNG and PDF bytes (no timestamps in the output)."""
        outputs = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as d:
                fig, ax = plt.subplots()
                ax.plot([0, 1], [0, 1], color=figstyle.c_exp)
                png, pdf = figstyle.save(fig, Path(d) / "fig")
                outputs.append((Path(png).read_bytes(), Path(pdf).read_bytes()))
        assert outputs[0] == outputs[1]
