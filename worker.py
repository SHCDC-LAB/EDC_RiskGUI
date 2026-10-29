"""Local adapter for the byte-preserved V10B.1 predictor and manuscript screen."""
from pathlib import Path
import argparse
import csv
import hashlib
import html
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent
RAW_NAME = 'endpoint_level_full41_v10b_external_gated.csv'
GROUPS = {'E': '雌激素', 'A': '雄激素', 'T': '甲状腺相关', 'S': '类固醇生成/转化'}
DOMAINS = {'in_domain': '适用域内', 'borderline_domain': '适用域边界',
           'out_of_domain': '适用域外', 'observed_parent_match': '数据库母体结构匹配',
           'invalid': '无效结构', 'not_available': '未提供'}
REQUIRED = {'numpy': '2.3.5', 'pandas': '2.2.3', 'scikit-learn': '1.8.0',
            'lightgbm': '4.6.0', 'rdkit': '2025.9.4', 'joblib': '1.5.3'}


def emit(message, **values):
    print(json.dumps({'message': message, **values}, ensure_ascii=False), flush=True)


def check_assets():
    for name, digest in json.loads((ROOT / 'asset_manifest.json').read_text()).items():
        file = ROOT / name
        if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != digest:
            raise RuntimeError('模型文件缺失或被改动，请重新解压完整包：' + name)


def check_environment():
    found = {name: importlib.metadata.version(name) for name in REQUIRED}
    different = [f'{name}={found[name]}（需要 {version}）' for name, version in REQUIRED.items()
                 if found[name] != version]
    if different:
        raise RuntimeError('依赖版本不匹配，请运行首次安装脚本：' + '；'.join(different))
    return {'python': platform.python_version(), 'platform': platform.platform(), **found}


def read_input(path, smiles_col='SMILES', name_col='Name', cas_col='CAS'):
    """Preserve identifiers as strings, including leading zeroes and literal NA."""
    raw = Path(path).read_bytes()
    encodings = ['utf-16'] if raw.startswith((b'\xff\xfe', b'\xfe\xff')) else ['utf-8-sig', 'gb18030']
    for encoding in encodings:
        try:
            text = raw.decode(encoding)
            break
        except UnicodeError:
            continue
    else:
        raise ValueError('无法读取文件编码，请另存为 UTF-8 CSV。')
    delimiter = '\t' if Path(path).suffix.lower() == '.tsv' else ','
    reader = csv.DictReader(text.splitlines(keepends=True), delimiter=delimiter)
    if not reader.fieldnames or smiles_col not in reader.fieldnames:
        raise ValueError('缺少结构列 ' + smiles_col + '；CSV请用英文逗号分隔，TSV请使用.tsv后缀。')
    if len(reader.fieldnames) != len(set(reader.fieldnames)):
        raise ValueError('文件存在重复列名，请先更正。')
    result = []
    for row in reader:
        if None in row:
            raise ValueError('某行的字段数超过表头，请检查逗号或引号。')
        if not any(str(v or '').strip() for v in row.values()):
            continue
        i = len(result) + 1
        result.append({'input_row_id': i, 'Name': str(row.get(name_col) or '').strip() or f'化合物{i}',
                       'SMILES': str(row.get(smiles_col) or '').strip(),
                       'CAS': str(row.get(cas_col) or '').strip()})
    if not result:
        raise ValueError('输入中没有化合物。')
    if len(result) > 2000:
        raise ValueError('每次最多处理2000行，请拆分文件后运行。')
    return result


def write_csv(path, rows, fields=None, protect=False):
    fields = fields or list(rows[0])
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            clean = {key: row.get(key, '') for key in fields}
            if protect:
                clean = {k: "'" + v if isinstance(v, str) and v.startswith(('=', '+', '-', '@', '\t', '\r'))
                         else v for k, v in clean.items()}
            w.writerow(clean)


def number(value):
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except (ValueError, TypeError):
        return None


def classify(row, metric):
    strict = number(row.get('observed_parent_pAC50_strict_v10'))
    broader = number(row.get('observed_parent_pAC50_v10'))
    p = number(row.get('activity_probability_calibrated_v10'))
    database = strict is not None and strict >= 5.5
    model = (strict is None and broader is None and p is not None and p >= .65
             and row['joint_AD_v10'] == 'in_domain'
             and metric['roc_auc'] >= .70 and metric['pr_auc'] > metric['active_fraction']
             and not metric['held'])
    if database:
        return '数据库证据', '严格质控数据 pAC50≥5.5'
    if model:
        return '模型候选', '概率≥0.65；适用域内；终点性能达标'
    if strict is not None:
        return '未入选', '严格质控数据未达到pAC50阈值'
    if broader is not None:
        return '待核查', '有较宽质控数据；不作为无实测数据的模型候选'
    reasons = []
    if metric['held']:
        reasons.append('该终点暂停模型候选筛选')
    if metric['roc_auc'] < .70 or metric['pr_auc'] <= metric['active_fraction']:
        reasons.append('终点性能未达到筛选门槛')
    if row['joint_AD_v10'] != 'in_domain':
        reasons.append(DOMAINS.get(row['joint_AD_v10'], row['joint_AD_v10']))
    if p is None or p < .65:
        reasons.append('活性概率未达到0.65')
    return '未入选', '；'.join(reasons)


def decorate(raw_rows, inputs, metrics):
    by_id = {str(r['input_row_id']): r for r in inputs}
    output = []
    for row in raw_rows:
        ep = row['endpoint_key']
        if ep not in metrics:
            continue
        m = metrics[ep]
        src = by_id[row['input_row_id']]
        status, reason = classify(row, m)
        output.append({'输入序号': int(row['input_row_id']), '化合物': src['Name'], 'CAS': src['CAS'],
                       'EATS组': GROUPS[m['group']], '终点': ep, '筛选结果': status,
                       '活性概率': number(row['activity_probability_calibrated_v10']),
                       '数据库pAC50_严格质控': number(row['observed_parent_pAC50_strict_v10']),
                       '数据库pAC50_较宽质控': number(row['observed_parent_pAC50_v10']),
                       '预测pAC50': number(row['predicted_pAC50_v10']),
                       '预测pAC50_90%区间下限': number(row['pAC50_interval90_lower_v10']),
                       '预测pAC50_90%区间上限': number(row['pAC50_interval90_upper_v10']),
                       '效力模型用途': row['pAC50_reliability_v10'],
                       '适用域': DOMAINS.get(row['joint_AD_v10'], row['joint_AD_v10']),
                       '证据来源': row['evidence_origin_v10'], '判断依据': reason,
                       '终点ROC_AUC': m['roc_auc'], '终点PR_AUC': m['pr_auc'],
                       '训练集阳性比例': m['active_fraction'], '终点暂停模型筛选': m['held'],
                       '标准化母体SMILES': row['parent_smiles_v10']})
    return output


def make_summary(inputs, detail, rejected):
    by_id = {}
    for row in detail:
        by_id.setdefault(row['输入序号'], []).append(row)
    summaries = []
    for src in inputs:
        i = src['input_row_id']
        rows = by_id.get(i, [])
        db = [r for r in rows if r['筛选结果'] == '数据库证据']
        mo = [r for r in rows if r['筛选结果'] == '模型候选']
        if i in rejected:
            state = '输入无效'
        elif db and mo:
            state = '数据库证据＋模型候选'
        elif db:
            state = '数据库证据'
        elif mo:
            state = '模型候选'
        else:
            state = '未达到本轮筛选条件'
        summaries.append({'输入序号': i, '化合物': src['Name'], 'CAS': src['CAS'], '结果': state,
                          '数据库入选终点数': len(db), '模型入选终点数': len(mo),
                          '关注终点': '；'.join(r['终点'] for r in db + mo),
                          '输入SMILES': src['SMILES'],
                          '标准化母体SMILES': rows[0]['标准化母体SMILES'] if rows else '',
                          '输入备注': rejected.get(i, src.get('note', ''))})
    return summaries


def write_report(out, summaries, details):
    fields = ['输入序号', '化合物', '结果', '数据库入选终点数', '模型入选终点数', '关注终点']
    esc = lambda x: html.escape(str(x if x is not None else ''))
    def table(rows, columns):
        return '<table><tr>' + ''.join('<th>' + esc(c) + '</th>' for c in columns) + '</tr>' + ''.join(
            '<tr>' + ''.join('<td>' + esc(r.get(c)) + '</td>' for c in columns) + '</tr>' for r in rows) + '</table>'
    hits = [r for r in details if r['筛选结果'] in ('数据库证据', '模型候选')]
    doc = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>EDC-AOPRisk 结果报告</title>
<style>body{font:15px/1.7 system-ui,sans-serif;max-width:1250px;margin:40px auto;padding:0 24px;color:#1e293b}h1{font-size:28px}h2{font-size:20px;margin-top:32px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border:1px solid #dbe1e8;padding:9px;text-align:left;word-break:break-word}th{background:#f1f5f9}p{color:#475569}@media print{body{margin:0}tr{break-inside:avoid}}</style>
<h1>EDC-AOPRisk · 内分泌活性筛选结果</h1><p>模型 V10B.1 ｜界面 1.0.2 ｜论文筛选规则 2026-10-28</p>
<p>以全部17个EATS终点筛选；24个辅助终点保留于原始输出。数据库证据指数据库中匹配母体结构的体外试验记录；模型候选属于后续试验线索。未入选不代表无活性或无风险。</p>
<h2>化合物汇总</h2>'''
    doc += table(summaries, fields)
    doc += '<h2>入选的化合物–终点配对</h2>' + table(hits, ['输入序号', '化合物', 'EATS组', '终点', '筛选结果', '活性概率', '数据库pAC50_严格质控', '适用域'])
    doc += '<h2>结果阅读</h2><p>pAC50 = −log10(AC50，单位mol/L)。预测pAC50仅作有相应性能支持时的辅助解释；概率、效力和健康风险是不同概念。对盐类按原模型保留主要片段并中和，详见输入备注和标准化母体结构。完整终点结果、未入选原因、效力区间及冻结原始输出均在同一目录。输入无效的行不运行预测。</p></html>'
    (out / '结果报告.html').write_text(doc, encoding='utf-8')


def run(request):
    started = time.time()
    emit('检查模型完整性和运行环境')
    check_assets()
    environment = check_environment()
    source = Path(request['input'])
    inputs = read_input(source, request.get('smiles_col', 'SMILES'), request.get('name_col', 'Name'), request.get('cas_col', 'CAS'))
    output = Path(request['output']).resolve()
    output.mkdir(parents=True, exist_ok=False)
    from runtime_guard import prepare_runtime
    runtime_validation = prepare_runtime(preflight=True, diagnostic_path=output/'运行环境诊断.json')
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.error')
    # Windows has a distinct process group so cancellation also stops the frozen child.
    rejected = {}; valid = []
    chooser = rdMolStandardize.LargestFragmentChooser(); uncharger = rdMolStandardize.Uncharger()
    for src in inputs:
        m = Chem.MolFromSmiles(src['SMILES']) if src['SMILES'] else None
        if m is None or not m.GetNumHeavyAtoms() or any(a.GetAtomicNum() == 0 for a in m.GetAtoms()):
            rejected[src['input_row_id']] = 'SMILES为空、无效或包含未定义原子；未运行预测'
            continue
        try:
            m = rdMolStandardize.Cleanup(m); m = chooser.choose(m); m = uncharger.uncharge(m); Chem.SanitizeMol(m)
        except Exception:
            rejected[src['input_row_id']] = '原模型结构标准化失败；未运行预测'
            continue
        notes = []
        if '.' in src['SMILES']:
            notes.append('多片段结构：原模型取主要片段并中和；请核对母体结构')
        if m.GetNumHeavyAtoms() > 60:
            notes.append('母体重原子数>60：升级模型2D描述符按原流程填补，需谨慎解释')
        src['note'] = '；'.join(notes)
        valid.append(src)
    raw_rows = []
    chunk_size = 100
    env = os.environ.copy()
    env.pop('RDBASE', None)
    env.update(PYTHONUTF8='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='1')
    engine = ROOT / 'predict_portable.py'
    raw_dir = output / '原始模型输出'; raw_dir.mkdir()
    with tempfile.TemporaryDirectory(prefix='edc_work_') as temp:
        work = Path(temp)
        for k in range(0, len(valid), chunk_size):
            chunk = valid[k:k + chunk_size]
            emit(f'正在预测 {k + 1}–{k + len(chunk)} / {len(valid)} 个有效结构', done=k, total=len(valid))
            chunk_input = work / f'input_{k}.csv'
            # The original CSV reader performs numeric type inference on metadata.
            # Use row tokens here, then restore exact user names/CAS in every output.
            engine_inputs = [{**r, 'Name': 'row_' + str(r['input_row_id']), 'CAS': 'metadata_not_used'} for r in chunk]
            write_csv(chunk_input, engine_inputs, ['input_row_id', 'Name', 'CAS', 'SMILES'])
            chunk_out = work / f'chunk_{k}'
            with (output / '运行日志.txt').open('a', encoding='utf-8') as log:
                process = subprocess.run([sys.executable, str(engine), '--input', str(chunk_input), '--output-dir', str(chunk_out)],
                                         stdout=log, stderr=log, env=env)
            diagnostics_dir = output/'特征诊断'; diagnostics_dir.mkdir(exist_ok=True)
            for diagnostic in chunk_out.glob('*.json'):
                import shutil
                shutil.copy2(diagnostic, diagnostics_dir/f'{k}_{diagnostic.name}')
            if process.returncode:
                raise RuntimeError('预测未完成，请查看输出目录中的运行日志。')
            by_id = {str(r['input_row_id']): r for r in chunk}
            for file in chunk_out.glob('*.csv'):
                with file.open(encoding='utf-8-sig', newline='') as f:
                    rows = list(csv.DictReader(f))
                for r in rows:
                    r['Name'] = by_id[r['input_row_id']]['Name']; r['CAS'] = by_id[r['input_row_id']]['CAS']
                dest = raw_dir / file.name
                first = not dest.exists()
                with dest.open('a', encoding='utf-8-sig' if first else 'utf-8', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0]))
                    if first: w.writeheader()
                    w.writerows(rows)
                if file.name == RAW_NAME:
                    raw_rows.extend(rows)
    metrics = json.loads((ROOT / 'screening_metrics.json').read_text())
    details = decorate(raw_rows, valid, metrics)
    assert len(details) == len(valid) * 17
    summary = make_summary(inputs, details, rejected)
    write_csv(output / '化合物汇总.csv', summary, protect=True)
    detail_fields = list(details[0]) if details else ['输入序号', '化合物', '终点', '筛选结果']
    write_csv(output / '全部EATS终点.csv', details, detail_fields, protect=True)
    hits = [r for r in details if r['筛选结果'] in ('数据库证据', '模型候选')]
    write_csv(output / '入选终点配对.csv', hits, detail_fields, protect=True)
    write_csv(output / '输入检查.csv', [{'输入序号': r['input_row_id'], '化合物': r['Name'],
              '状态': '无效' if r['input_row_id'] in rejected else '有效',
              '备注': rejected.get(r['input_row_id'], r.get('note', ''))} for r in inputs], protect=True)
    write_report(output, summary, details)
    result = {'status': 'completed', 'version': 'GUI-1.0.2 / frozen-V10B.1', 'runtime_validation': runtime_validation,
              'input_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'environment': environment, 'n_input': len(inputs), 'n_valid': len(valid),
              'n_invalid': len(rejected), 'n_database_pairs': sum(r['筛选结果'] == '数据库证据' for r in hits),
              'n_model_pairs': sum(r['筛选结果'] == '模型候选' for r in hits),
              'n_priority_compounds': sum(r['数据库入选终点数'] + r['模型入选终点数'] > 0 for r in summary),
              'seconds': round(time.time() - started, 2), 'summary': summary, 'details': details}
    (output / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    emit('完成', output=str(output), n_valid=len(valid), n_invalid=len(rejected))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    args = parser.parse_args()
    try:
        run(json.loads(Path(args.request).read_text(encoding='utf-8')))
    except Exception as error:
        emit(str(error), error=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
