from pathlib import Path
import json
from worker import check_assets, check_environment

def main():
    env = check_environment()
    check_assets()
    from runtime_guard import prepare_runtime
    runtime = prepare_runtime(preflight=True, diagnostic_path=Path(__file__).resolve().parent/'Startup_Runtime.json')
    import joblib
    root = Path(__file__).resolve().parent
    count = 0
    for file in (root/'engine/models').rglob('*.pkl'):
        bundle = joblib.load(file)
        assert 'endpoint_key' in bundle and 'model' in bundle, file
        count += 1
    assert count == 147, count
    print(json.dumps({'status':'PASS','model_bundles':count,'environment':env,'runtime_validation':runtime}, indent=2))

if __name__ == '__main__':
    main()
