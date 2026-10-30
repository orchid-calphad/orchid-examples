"""Tests for the Cu-Zr workflow: provenance records, locked inputs, stage order, helpers and the stage runner."""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path
from unittest import TestCase, mock

import jsonschema

root = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("workflow", root / "scripts/workflow.py")
workflow_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(workflow_mod)
_spec = importlib.util.spec_from_file_location("helpers", root / "scripts/workflow_helpers.py")
helpers_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(helpers_mod)


class TestProvenance(TestCase):
    """Provenance records of the archived TDB files."""

    def test_provenance_schema_and_exact_tdb_hashes(self):
        """Provenance validates against its schema and every recorded TDB digest matches its file."""
        data = json.loads((root / "tdb/model-provenance.json").read_text())
        schema = json.loads((root / "tdb/model-provenance.schema.json").read_text())
        jsonschema.Draft202012Validator(schema).validate(data)
        for r in data["records"]:
            self.assertEqual(workflow_mod.sha(root / r["path"]), r["sha256"])
            self.assertIn(r["generation_method"], r["tags"])
        final = next(r for r in data["records"] if r["generation_method"] == "phaseforge_plus")
        self.assertEqual(final["workflow_history"], ["phaseforge", "phaseforge_plus"])
        self.assertEqual(final["potential"]["name"], "GRACE-2L-OMAT")
        self.assertEqual(len(final["potential"]["checkpoint"]["files"]), 6)

    def test_schema_rejects_malformed_digest(self):
        """The schema rejects a record whose sha256 is not a valid digest."""
        data = json.loads((root / "tdb/model-provenance.json").read_text())
        data["records"][0]["sha256"] = "made-up"
        schema = json.loads((root / "tdb/model-provenance.schema.json").read_text())
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(data, schema)


class TestLockedInputs(TestCase):
    """Locked workflow inputs."""

    def test_all_workflow_inputs_are_locked(self):
        """Every locked input verifies, and changing one recorded digest makes verification fail."""
        manifest = json.loads((root / "workflow/cuzr.json").read_text())
        self.assertEqual(workflow_mod.verify(manifest), [])
        changed = copy.deepcopy(manifest)
        changed["input_sha256"]["tdb/CuZr-PhaseForge-final.tdb"] = "0" * 64
        self.assertTrue(workflow_mod.verify(changed))


class TestStageOrder(TestCase):
    """Dependency ordering of the workflow stages."""

    def test_dependencies_precede_dependents(self):
        """Every stage in an evaluation plan appears after all of its dependencies."""
        manifest = json.loads((root / "workflow/cuzr.json").read_text())
        stages = workflow_mod.order(manifest, ["evaluate_s3"])
        positions = {s["id"]: i for i, s in enumerate(stages)}
        for s in stages:
            for dep in s["depends_on"]:
                self.assertLess(positions[dep], positions[s["id"]])

    def test_cyclic_dependency_is_rejected(self):
        """A stage that depends on itself raises a cycle error."""
        manifest = json.loads((root / "workflow/cuzr.json").read_text())
        manifest["stages"][0]["depends_on"] = [manifest["stages"][0]["id"]]
        with self.assertRaisesRegex(ValueError, "Cyclic"):
            workflow_mod.order(manifest, ["cached_gate"])


class TestFullScriptImports(TestCase):
    """Imports between the full-workflow scripts and their shared path module."""

    def test_full_scripts_import_only_existing_path_symbols(self):
        """Every name imported from _paths is defined in _paths.py."""
        tree = ast.parse((root / "scripts/full/_paths.py").read_text())
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                names.add(node.name)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                names.add(node.id)
        for p in (root / "scripts/full").glob("*.py"):
            for node in ast.walk(ast.parse(p.read_text())):
                if isinstance(node, ast.ImportFrom) and node.module == "_paths":
                    self.assertLessEqual({a.name for a in node.names}, names, p)


class TestHelpers(TestCase):
    """Preparation helpers and the output-snapshot utilities."""

    def setUp(self):
        """Create a temporary directory that is removed after each test."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp_path = Path(tmp.name)

    def test_b2_model_preparation_matches_archived_variant(self):
        """The B2 variant written by the helper is byte-identical to the archived TDB."""
        (self.tmp_path / "tdb").mkdir()
        shutil.copy2(root / "tdb/CuZr-PhaseForge-MLIP.tdb", self.tmp_path / "tdb")
        helpers_mod.b2_variant(self.tmp_path)
        written = (self.tmp_path / "tdb/CuZr-PhaseForge-MLIP-B2b.tdb").read_bytes()
        self.assertEqual(written, (root / "tdb/CuZr-PhaseForge-MLIP-B2b.tdb").read_bytes())

    def test_missing_outputs_raise(self):
        """A required output pattern that matches no file raises ValueError."""
        with self.assertRaisesRegex(ValueError, "No artifact"):
            workflow_mod.snapshot(["{run}/missing/*.json"], {"run": self.tmp_path})

    def test_placeholder_values_are_not_shell_expanded(self):
        """Placeholder values containing shell syntax are substituted literally."""
        value = "value with spaces; $(do-not-run)"
        self.assertEqual(workflow_mod.expand("{item}", {"item": value}), value)


class TestRunner(TestCase):
    """Execution of a single stage: receipts, resume and modified-output detection."""

    def setUp(self):
        """Create a temporary directory that is removed after each test."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp_path = Path(tmp.name)

    def test_execute_receipt_resume_and_detect_modified_output(self):
        """A run writes a receipt, resumes from it, and refuses a stage whose output changed."""
        source = self.tmp_path / "source"
        source.mkdir()
        out = self.tmp_path / "run"
        (source / "requirements.txt").write_text("")
        script = source / "write.py"
        script.write_text("import pathlib,sys;pathlib.Path(sys.argv[1]).write_text('ok')")
        m = {
            "id": "test",
            "input_sha256": {},
            "potential_manifest": "unused",
            "environments": {"quick": "requirements.txt"},
            "stages": [
                {
                    "id": "test",
                    "environment": "quick",
                    "depends_on": [],
                    "inputs": [],
                    "outputs": ["{run}/result.txt"],
                    "requires_potential": False,
                    "requires_orchid": False,
                    "commands": [["{python}", "{example}/write.py", "{run}/result.txt"]],
                }
            ],
        }
        manifest = source / "manifest.json"
        manifest.write_text(json.dumps(m))
        argv = ["workflow", "run", "--manifest", str(manifest), "--stage", "test", "--run-dir", str(out)]
        with (
            mock.patch.object(workflow_mod, "example", source),
            mock.patch.object(workflow_mod, "verify", lambda *a, **k: []),
        ):
            with mock.patch.object(sys, "argv", argv):
                self.assertEqual(workflow_mod.main(), 0)
            receipt = json.loads((out / "receipts/test.json").read_text())
            self.assertEqual(receipt["outputs"][str(out / "result.txt")], workflow_mod.sha(out / "result.txt"))
            with mock.patch.object(sys, "argv", [*argv, "--resume"]):
                self.assertEqual(workflow_mod.main(), 0)
            (out / "result.txt").write_text("changed")
            with (
                mock.patch.object(sys, "argv", [*argv, "--resume"]),
                self.assertRaisesRegex(ValueError, "Stage inputs/outputs changed"),
            ):
                workflow_mod.main()
