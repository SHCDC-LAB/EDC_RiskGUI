"""Guarded entrypoint; the frozen scientific prediction script stays unchanged."""
from pathlib import Path
import argparse
import runpy
from runtime_guard import prepare_runtime, audit_fragments

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--smiles-col', default='SMILES')
    args, _ = parser.parse_known_args()
    target = Path(args.output_dir); target.mkdir(parents=True, exist_ok=True)
    prepare_runtime(preflight=False, diagnostic_path=target/'runtime_child_diagnostic.json')
    audit_fragments(args.input, target, smiles_col=args.smiles_col)
    runpy.run_path(str(Path(__file__).resolve().parent/'engine/scripts/predict_new_smiles_v10b_audit_frozen.py'), run_name='__main__')
