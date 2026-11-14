"""Verify frozen runtime plus transitive local helpers before confirmation."""
import argparse
import ast
import hashlib
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def helpers(filename, seen=None):
    seen = set() if seen is None else seen
    if filename in seen:
        return seen
    seen.add(filename)
    tree = ast.parse((ROOT / filename).read_text(encoding='utf-8-sig'))
    for node in ast.walk(tree):
        names = ([n.name for n in node.names] if isinstance(node, ast.Import)
                 else [node.module or ''] if isinstance(node, ast.ImportFrom) else [])
        for name in names:
            local = name.split('.')[0] + '.py'
            if (ROOT / local).is_file():
                helpers(local, seen)
    return seen


def verify_runtime():
    design = json.loads((ROOT / 'confirmation27_design.json').read_text())
    manifest_file = ROOT / 'confirmation27_runtime_manifest.json'
    assert sha(manifest_file) == design['runtime_manifest_sha256']
    assert sha(ROOT / 'model_hashes.json') == design['model_manifest_sha256']
    for name, expected in json.loads(manifest_file.read_text()).items():
        assert sha(ROOT / 'confirmation27_bin' / name) == expected, name
    for key, name in [('tasks_sha256', 'confirmation27_heldout192.jsonl'),
                      ('settings_sha256', 'confirmation27_settings.json'),
                      ('runner_sha256', 'paired_generation.py')]:
        assert sha(ROOT / name) == design[key], key
    return design


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    verify_runtime()
    destination = ROOT / 'confirmation27_helpers'
    supplemental = ROOT / 'confirmation27_helper_manifest.json'
    if args.freeze:
        assert not (ROOT / 'confirmation27_manifest.json').exists(), 'Inference already launched'
        destination.mkdir()
        files = sorted(helpers('paired_generation.py'))
        for name in files:
            shutil.copy2(ROOT / name, destination / name)
        doc = {'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
               'purpose': 'Additional provenance before held-out inference; no experimental changes',
               'design_sha256': sha(ROOT / 'confirmation27_design.json'),
               'runtime_manifest_sha256': sha(ROOT / 'confirmation27_runtime_manifest.json'),
               'files': {name: sha(ROOT / name) for name in files},
               'python': sys.version, 'python_executable': sys.executable,
               'platform': platform.platform()}
        with supplemental.open('x') as f:
            json.dump(doc, f, indent=2)
    doc = json.loads(supplemental.read_text())
    assert sha(ROOT / 'confirmation27_design.json') == doc['design_sha256']
    for name, expected in doc['files'].items():
        assert sha(ROOT / name) == expected, ('live helper changed', name)
        assert sha(destination / name) == expected, ('snapshot changed', name)
    print(json.dumps({'preflight': 'passed', 'frozen_runtime_files': 33,
                      'transitive_helpers': len(doc['files']),
                      'helper_manifest_sha256': sha(supplemental)}))


if __name__ == '__main__':
    main()
