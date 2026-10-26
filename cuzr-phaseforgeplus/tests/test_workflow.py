import ast
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import pytest
import jsonschema

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('workflow',ROOT/'scripts/workflow.py');W=importlib.util.module_from_spec(spec);spec.loader.exec_module(W)
spec=importlib.util.spec_from_file_location('helpers',ROOT/'scripts/workflow_helpers.py');H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H)


def test_provenance_schema_and_exact_tdb_hashes():
    data=json.loads((ROOT/'tdb/model-provenance.json').read_text());schema=json.loads((ROOT/'tdb/model-provenance.schema.json').read_text())
    jsonschema.Draft202012Validator(schema).validate(data)
    for r in data['records']:
        assert W.sha(ROOT/r['path'])==r['sha256']
        assert r['generation_method'] in r['tags']
    final=next(r for r in data['records'] if r['generation_method']=='phaseforge_plus')
    assert final['workflow_history']==['phaseforge','phaseforge_plus']
    assert final['potential']['name']=='GRACE-2L-OMAT' and len(final['potential']['checkpoint']['files'])==6


def test_schema_rejects_malformed_digest():
    d=json.loads((ROOT/'tdb/model-provenance.json').read_text());d['records'][0]['sha256']='made-up'
    with pytest.raises(jsonschema.ValidationError):jsonschema.validate(d,json.loads((ROOT/'tdb/model-provenance.schema.json').read_text()))


def test_all_workflow_inputs_are_locked():
    m=json.loads((ROOT/'workflow/cuzr.json').read_text());assert W.verify(m)==[]
    changed=copy.deepcopy(m);changed['input_sha256']['tdb/CuZr-PhaseForge-final.tdb']='0'*64
    assert W.verify(changed)


def test_dependency_order_and_cycles():
    m=json.loads((ROOT/'workflow/cuzr.json').read_text());stages=W.order(m,['evaluate_s3']);positions={s['id']:i for i,s in enumerate(stages)}
    for s in stages:
        for dep in s['depends_on']:assert positions[dep]<positions[s['id']]
    m['stages'][0]['depends_on']=[m['stages'][0]['id']]
    with pytest.raises(ValueError,match='Cyclic'):W.order(m,['cached_gate'])


def test_full_scripts_import_only_existing_path_symbols():
    tree=ast.parse((ROOT/'scripts/full/_paths.py').read_text());names=set()
    for node in ast.walk(tree):
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)):names.add(node.name)
        if isinstance(node,ast.Name) and isinstance(node.ctx,ast.Store):names.add(node.id)
    for p in (ROOT/'scripts/full').glob('*.py'):
        for node in ast.walk(ast.parse(p.read_text())):
            if isinstance(node,ast.ImportFrom) and node.module=='_paths':
                assert {a.name for a in node.names}<=names,p


def test_b2_model_preparation_matches_archived_variant(tmp_path):
    (tmp_path/'tdb').mkdir();shutil.copy2(ROOT/'tdb/CuZr-PhaseForge-MLIP.tdb',tmp_path/'tdb')
    H.b2_variant(tmp_path)
    assert (tmp_path/'tdb/CuZr-PhaseForge-MLIP-B2b.tdb').read_bytes()==(ROOT/'tdb/CuZr-PhaseForge-MLIP-B2b.tdb').read_bytes()


def test_missing_outputs_fail_and_arguments_are_not_shell_expanded(tmp_path):
    with pytest.raises(ValueError,match='No artifact'):W.snapshot(['{run}/missing/*.json'],{'run':tmp_path})
    assert W.expand('{item}',{'item':'value with spaces; $(do-not-run)'})=='value with spaces; $(do-not-run)'


def test_execute_receipt_resume_and_detect_modified_output(tmp_path,monkeypatch):
    import sys
    source=tmp_path/'source';source.mkdir();out=tmp_path/'run'
    (source/'requirements.txt').write_text('')
    script=source/'write.py';script.write_text("import pathlib,sys;pathlib.Path(sys.argv[1]).write_text('ok')")
    m={'id':'test','input_sha256':{},'potential_manifest':'unused','environments':{'quick':'requirements.txt'},
       'stages':[{'id':'test','environment':'quick','depends_on':[],'inputs':[],
                  'outputs':['{run}/result.txt'],'requires_potential':False,'requires_orchid':False,
                  'commands':[['{python}','{example}/write.py','{run}/result.txt']]}]}
    manifest=source/'manifest.json';manifest.write_text(json.dumps(m))
    monkeypatch.setattr(W,'EXAMPLE',source);monkeypatch.setattr(W,'verify',lambda *a,**k:[])
    argv=['workflow','run','--manifest',str(manifest),'--stage','test','--run-dir',str(out)]
    monkeypatch.setattr(sys,'argv',argv);assert W.main()==0
    receipt=json.loads((out/'receipts/test.json').read_text());assert receipt['outputs'][str(out/'result.txt')]==W.sha(out/'result.txt')
    monkeypatch.setattr(sys,'argv',argv+['--resume']);assert W.main()==0
    (out/'result.txt').write_text('changed')
    with pytest.raises(ValueError,match='Stage inputs/outputs changed'):W.main()
