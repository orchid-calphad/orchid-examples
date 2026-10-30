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
            (Path(d) / "espei" / "index.csv").write_text("a\n", encoding="utf-8")
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
                (folder / name).write_text("{}", encoding="utf-8")
            assert orchid_data.listdir("espei/x", "0" * 40) == ["espei/x/a.json", "espei/x/b.json"]

    def test_local_root_is_none_when_unset(self):
        """Without ORCHID_DATA the local checkout is not used."""
        with mock.patch.dict(os.environ, {}):
            os.environ.pop("ORCHID_DATA", None)
            assert orchid_data.local_root() is None


COMMIT = "a" * 40  # any orchid-data commit: the offline tests never reach GitHub


def _offline(cache: str, url: str | None = None):
    """Return a patch of os.environ with a private cache, no local checkout and an optional download template.

    Args:
        cache (str): Directory used as ORCHID_CACHE.
        url (str | None): ORCHID_DATA_URL template; None keeps the variable unset.

    Returns:
        mock._patch: Context manager that patches os.environ.
    """
    env = {k: v for k, v in os.environ.items() if k not in ("ORCHID_DATA", "ORCHID_DATA_URL", "ORCHID_CACHE")}
    env["ORCHID_CACHE"] = cache
    if url is not None:
        env["ORCHID_DATA_URL"] = url
    return mock.patch.dict(os.environ, env, clear=True)


class TestRemoteRelease(TestCase):
    """Tests for release-file lookup without ORCHID_DATA: cache hits, downloads and the espei index (offline)."""

    def test_cache_dir_honours_orchid_cache(self):
        """ORCHID_CACHE selects the download cache directory."""
        with tempfile.TemporaryDirectory() as d, _offline(d):
            assert orchid_data.cache_dir() == Path(d)

    def test_cached_file_is_returned_without_download(self):
        """A file already in the cache for the commit is returned and the download URL is never used."""
        with tempfile.TemporaryDirectory() as d, _offline(d, url="file:///nonexistent-orchid/{commit}/{path}"):
            target = Path(d) / COMMIT / "espei" / "index.csv"
            target.parent.mkdir(parents=True)
            target.write_text("path\n", encoding="utf-8")
            assert orchid_data.path("espei/index.csv", COMMIT) == target

    def test_download_is_written_to_the_cache_once(self):
        """A downloaded file is stored under <cache>/<commit>/<relpath> with no .part leftover, and reused afterwards."""
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as cache:
            (Path(src) / "espei").mkdir()
            source = Path(src) / "espei" / "index.csv"
            source.write_bytes(b"path\nx.json\n")
            url = Path(src).as_uri() + "/{path}"
            with _offline(cache, url=url):
                got = orchid_data.path("espei/index.csv", COMMIT)
                assert got == Path(cache) / COMMIT / "espei" / "index.csv"
                assert got.read_bytes() == b"path\nx.json\n"
                assert not got.with_suffix(".csv.part").exists()
                source.unlink()
                assert orchid_data.path("espei/index.csv", COMMIT) == got

    def test_failed_download_names_the_remedy(self):
        """A download that fails raises RuntimeError that names the commit and the ORCHID_DATA remedy."""
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as empty:
            url = Path(empty).as_uri() + "/missing/{commit}/{path}"
            with _offline(d, url=url), self.assertRaises(RuntimeError) as ctx:
                orchid_data.path("espei/index.csv", COMMIT)
            assert "ORCHID_DATA" in str(ctx.exception)
            assert COMMIT in str(ctx.exception)

    def test_remote_listdir_reads_the_espei_index(self):
        """Without a local checkout, listdir of an espei/ folder is answered from the commit's espei/index.csv."""
        with tempfile.TemporaryDirectory() as d, _offline(d):
            index = Path(d) / COMMIT / "espei" / "index.csv"
            index.parent.mkdir(parents=True)
            index.write_text(
                "path\nexperimental/CU-ZR/a.json\nexperimental/CU-ZR/b.csv\nexperimental/CU-ZR/c.json\ninvariants/d.json\n",
                encoding="utf-8",
            )
            assert orchid_data.listdir("espei/experimental/CU-ZR", COMMIT) == [
                "espei/experimental/CU-ZR/a.json",
                "espei/experimental/CU-ZR/c.json",
            ]

    def test_remote_listdir_rejects_folders_outside_espei(self):
        """A remote listing outside espei/ is refused with a RuntimeError that asks for ORCHID_DATA."""
        with tempfile.TemporaryDirectory() as d, _offline(d), self.assertRaisesRegex(RuntimeError, "ORCHID_DATA"):
            orchid_data.listdir("cuzr/other", COMMIT)


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
