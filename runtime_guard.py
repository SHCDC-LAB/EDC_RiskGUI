"""Restore the frozen fragment definitions and fail on runtime feature drift.

No model is refitted. The original predictor and its input feature order remain
unchanged. All changes below are confined to this Python process.
"""
from pathlib import Path
import csv
import hashlib
import json
import os
import platform
import sys

ROOT = Path(__file__).resolve().parent


def install_bertz_dtype_fix(info):
    """Backport RDKit's explicit-float entropy inputs (#8421 / #9072).

    Adapted from RDKit GraphDescriptors._CalculateEntropies, copyright
    Greg Landrum and Rational Discovery LLC, BSD license in runtime_resources.
    Keep the same formula and native entropy routine; only specify float64
    for the count arrays to avoid Windows/NumPy2's unsupported integer dtype.
    """
    import math
    import numpy as np
    from rdkit.Chem import GraphDescriptors
    from rdkit.ML.InfoTheory import entropy

    counts = np.array([4, 1, 12])
    info['entropy_dtype_probe'] = {
        'integer_dtype': str(counts.dtype),
        'integer_entropy': float(entropy.InfoEntropy(counts)),
        'float64_entropy': float(entropy.InfoEntropy(counts.astype(np.float64))),
        'python_reference_entropy': float(entropy.PyInfoEntropy(counts.astype(np.float64))),
    }

    def calculate_entropies(connectionDict, atomTypeDict, numAtoms):
        connectionList = list(connectionDict.values())
        totConnections = sum(connectionList)
        connectionIE = totConnections * (
            entropy.InfoEntropy(np.array(connectionList, dtype=np.float64))
            + math.log(totConnections) / math.log(2))
        atomTypeList = list(atomTypeDict.values())
        atomTypeIE = numAtoms * entropy.InfoEntropy(np.array(atomTypeList, dtype=np.float64))
        return atomTypeIE + connectionIE

    GraphDescriptors._CalculateEntropies = calculate_entropies
    info['bertz_entropy_input_dtype'] = 'float64'
    info['bertz_fix_reference'] = 'https://github.com/rdkit/rdkit/pull/9072'


def prepare_runtime(preflight=True, diagnostic_path=None):
    previous_rdbase = os.environ.pop('RDBASE', None)
    # RDConfig consults RDBASE before RDPaths. Do not inherit another program's
    # resource path or the parent process's automatically constructed RDBASE.
    import rdkit
    from rdkit import Chem, RDConfig, RDPaths
    data_candidates = [Path(rdkit.__file__).resolve().parent / 'Data', Path(RDPaths.RDDataDir)]
    data_dir = next((p for p in data_candidates if (p / 'FragmentDescriptors.csv').is_file()), None)
    if data_dir is not None:
        RDConfig.RDDataDir = str(data_dir)
    from rdkit.Chem import Descriptors, Fragments
    import numpy as np

    required = [r['descriptor_name'] for r in csv.DictReader(
        (ROOT / 'engine/data/v10a_rdkit_2d_descriptor_columns.csv').open(encoding='utf-8-sig'))]
    before = dict(Descriptors.descList)
    info = {'status': 'checking', 'gui_version': '1.0.2', 'python': platform.python_version(), 'platform': platform.platform(),
            'rdkit': rdkit.__version__, 'rdkit_module': str(Path(rdkit.__file__).resolve()),
            'inherited_RDBASE': previous_rdbase, 'resolved_RDDataDir': str(data_dir),
            'required_descriptor_count': len(required),
            'missing_before_repair': [n for n in required if n not in before],
            'fragments_before_repair': sum(n.startswith('fr_') for n in before)}
    try:
        resources = ROOT / 'runtime_resources'
        hashes = json.loads((resources / 'manifest.json').read_text())
        for name, digest in hashes.items():
            file = resources / name
            if hashlib.sha256(file.read_bytes()).hexdigest() != digest:
                raise RuntimeError('运行资源校验失败：' + name)
        install_bertz_dtype_fix(info)
        # Rebuild all 85 functions from the exact reference resource, with an
        # explicit encoding. Matching semantics equal RDKit Fragments.py.
        patterns = {}
        with (resources / 'FragmentDescriptors.csv').open(encoding='utf-8') as f:
            for line in f:
                if not line or line.startswith('#'):
                    continue
                fields = line.split('\t')
                if len(fields) < 3:
                    continue
                name = fields[0].replace('=', '_').replace('-', '_')
                pattern = Chem.MolFromSmarts(fields[2])
                if pattern is None or not pattern.GetNumAtoms():
                    raise RuntimeError('片段SMARTS定义无效：' + name)
                def count(molecule, countUnique=True, pattern=pattern):
                    return len(molecule.GetSubstructMatches(pattern, uniquify=countUnique))
                patterns[name] = count
        if len(patterns) != 85:
            raise RuntimeError('片段描述符数量异常：' + str(len(patterns)))
        for name, function in patterns.items():
            setattr(Fragments, name, function)
            setattr(Descriptors, name, function)
        # Avoid stale callable objects and duplicate names in the descriptor list.
        functions = {n: fn for n, fn in Descriptors.descList if not n.startswith('fr_')}
        functions.update(patterns)
        Descriptors.descList[:] = list(functions.items())
        Descriptors._descList = Descriptors.descList
        missing = [n for n in required if not callable(functions.get(n))]
        if missing:
            raise RuntimeError('缺少必需的分子描述符：' + ', '.join(missing))
        info.update(fragments_after_repair=85, missing_after_repair=missing,
                    fragment_definition_sha256=hashes['FragmentDescriptors.csv'])
        if preflight:
            check_reference(functions, required, info)
        info['status'] = 'PASS'
        return info
    except Exception as error:
        info.update(status='FAIL', error=str(error))
        raise RuntimeError('运行环境数值自检未通过，已停止预测：' + str(error)) from error
    finally:
        if diagnostic_path:
            Path(diagnostic_path).write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')


def check_reference(functions, names, info):
    import joblib
    import numpy as np
    from rdkit import Chem, DataStructs
    from rdkit.Chem import Descriptors, Crippen, Lipinski, rdFingerprintGenerator, rdMolDescriptors
    from rdkit.Chem.MolStandardize import rdMolStandardize
    reference = np.load(ROOT / 'runtime_resources/reference_features.npz')
    runtime = np.load(ROOT / 'engine/data/v10a_descriptor_runtime.npz')
    median, low, high = [runtime[k].astype(np.float32) for k in ['impute_median', 'clip_low', 'clip_high']]
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    chooser, uncharger = rdMolStandardize.LargestFragmentChooser(), rdMolStandardize.Uncharger()
    rows = list(csv.DictReader((ROOT / 'examples/demo_compounds.csv').open(encoding='utf-8-sig')))
    if len(rows) != 4:
        raise RuntimeError('内置自检示例已被修改；请在其他文件中编辑自己的输入。')
    xs, ds, failures = [], [], []
    for row in rows:
        mol = Chem.MolFromSmiles(row['SMILES'])
        mol = uncharger.uncharge(chooser.choose(rdMolStandardize.Cleanup(mol)))
        Chem.SanitizeMol(mol)
        bits = np.zeros(1024, np.float32); DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), bits)
        phys = np.array([Descriptors.MolWt(mol), Crippen.MolLogP(mol), rdMolDescriptors.CalcTPSA(mol),
            Lipinski.NumHDonors(mol), Lipinski.NumHAcceptors(mol), Lipinski.NumRotatableBonds(mol),
            mol.GetNumHeavyAtoms(), rdMolDescriptors.CalcNumRings(mol), rdMolDescriptors.CalcFractionCSP3(mol),
            Chem.GetFormalCharge(mol)], np.float32)
        xs.append(np.r_[bits, phys])
        values = []
        for name in names:
            try:
                v = float(functions[name](mol))
                if not np.isfinite(v):
                    raise ValueError('nonfinite')
                values.append(v)
            except Exception as error:
                failures.append({'chemical': row['Name'], 'descriptor': name, 'error': repr(error)})
                values.append(np.nan)
        values = np.asarray(values, np.float32)
        bad = ~np.isfinite(values); values[bad] = median[bad]
        ds.append(np.clip(values, low, high).astype(np.float32))
    x, d = np.asarray(xs), np.asarray(ds)
    info['reference_calculation_failures'] = failures
    if failures:
        raise RuntimeError('自检示例的描述符计算失败，不能使用默认值代替。')
    if not np.array_equal(x[:, :1024], reference['Xq'][:, :1024]):
        raise RuntimeError('Morgan指纹与参考环境不同。')
    if not np.allclose(x[:, 1024:], reference['Xq'][:, 1024:], atol=1e-6, rtol=1e-6):
        raise RuntimeError('基础理化描述符与参考环境不同。')
    close = np.isclose(d, reference['X2'], atol=1e-6, rtol=1e-6)
    info['descriptor_mismatches'] = [{'chemical': rows[i]['Name'], 'descriptor': names[j],
        'actual': float(d[i, j]), 'expected': float(reference['X2'][i, j])}
        for i, j in np.argwhere(~close)][:25]
    if not close.all():
        raise RuntimeError('扩展分子描述符与参考环境不同，详见运行环境诊断。')
    with (ROOT / 'examples/expected_demo_screen.csv').open(encoding='utf-8-sig') as f:
        expected = {(r['Name'], r['endpoint']): float(r['probability']) for r in csv.DictReader(f)}
    differences = []
    for ep in ['AR_AGONIST', 'AR_ANTAGONIST', 'ESR1_AGONIST']:
        bundle = joblib.load(ROOT / f'engine/models/activity_v10_hybrid/{ep}__activity_v9.pkl')
        indices = np.asarray(bundle.get('descriptor_indices', []), int)
        matrix = np.concatenate([x, d[:, indices]], axis=1) if len(indices) else x
        raw = bundle['model'].predict_proba(matrix)[:, 1]
        cal = bundle.get('calibrator')
        transformed = np.log(np.clip(raw, 1e-6, 1-1e-6) / (1-np.clip(raw, 1e-6, 1-1e-6))) \
            if bundle.get('calibration_input') == 'logit_raw_probability' else raw
        prob = cal.predict_proba(transformed.reshape(-1, 1))[:, 1] if cal is not None else raw
        for row, score in zip(rows, prob):
            ref = expected[(row['Name'], ep)]
            differences.append({'chemical': row['Name'], 'endpoint': ep, 'actual': float(score),
                                'expected': ref, 'absolute_difference': abs(float(score)-ref)})
    info['reference_predictions'] = differences
    info['maximum_probability_difference'] = max(r['absolute_difference'] for r in differences)
    if info['maximum_probability_difference'] > 1e-10:
        raise RuntimeError('3个升级模型的示例概率未通过参考数值核对。')
    info.update(reference_structures_checked=4, reference_descriptors_checked=len(names),
                reference_predictions_checked=12)


def audit_fragments(input_csv, output_dir, smiles_col='SMILES'):
    """Fail on fragment computation errors for valid small input molecules.

    The original greater-than-60-heavy-atoms policy is preserved, and reported.
    Other naturally undefined descriptors keep the original imputation policy.
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit.Chem.MolStandardize import rdMolStandardize
    chooser, uncharger = rdMolStandardize.LargestFragmentChooser(), rdMolStandardize.Uncharger()
    fragments = {n: f for n, f in Descriptors.descList if n.startswith('fr_')}
    if len(fragments) != 85:
        raise RuntimeError('片段描述符不完整，已停止预测。')
    with Path(input_csv).open(encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    issues = []
    for index, row in enumerate(rows, 1):
        mol = Chem.MolFromSmiles(row[smiles_col])
        mol = uncharger.uncharge(chooser.choose(rdMolStandardize.Cleanup(mol)))
        Chem.SanitizeMol(mol)
        if mol.GetNumHeavyAtoms() > 60:
            continue
        for name, function in fragments.items():
            try:
                function(mol)
            except Exception as error:
                issues.append({'input_row_id': row.get('input_row_id', str(index)), 'descriptor': name, 'error': repr(error)})
    if issues:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / 'fragment_errors.json').write_text(json.dumps(issues, ensure_ascii=False, indent=2), encoding='utf-8')
        raise RuntimeError('输入结构的片段特征计算失败，已停止；没有用默认值继续预测。')


if __name__ == '__main__':
    try:
        from worker import check_assets, check_environment
        check_assets()
        check_environment()
        result = prepare_runtime(preflight=True, diagnostic_path=ROOT/'Startup_Runtime.json')
        print(json.dumps({'status': result['status'], 'reference_predictions_checked': 12,
                          'maximum_probability_difference': result['maximum_probability_difference']}, indent=2))
    except Exception as error:
        print(str(error))
        sys.exit(1)
