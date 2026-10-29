"""Optional end-to-end numerical regression using four public example structures."""
from pathlib import Path
import csv, tempfile
from worker import run, number
root = Path(__file__).resolve().parent
with tempfile.TemporaryDirectory(prefix='edc_verify_') as temp:
    result = run({'input': str(root/'examples/demo_compounds.csv'), 'output': str(Path(temp)/'result')})
    actual = {(r['化合物'],r['终点']):r for r in result['details']}
    with (root/'examples/expected_demo_screen.csv').open(encoding='utf-8-sig') as f:
        expected = list(csv.DictReader(f))
    assert len(actual)==len(expected)==68
    for e in expected:
        r = actual[(e['Name'], e['endpoint'])]
        assert r['筛选结果']==e['status']
        for dst,src in [('活性概率','probability'),('数据库pAC50_严格质控','strict_pAC50')]:
            a,b=number(r[dst]),number(e[src])
            assert (a is None)==(b is None)
            if a is not None: assert abs(a-b)<1e-10, (e['Name'],e['endpoint'],dst,a,b)
    print('PASS: 4 compounds / 68 Core-EATS pairs; probabilities, strict potency and screen decisions match.')
