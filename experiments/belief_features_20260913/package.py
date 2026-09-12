"""Package review artifacts while leaving large raw data/checkpoints in place."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


def package(root, output):
    analysis = json.loads((root / 'analysis_final/synthesis.json').read_text())
    assert analysis['status'] == 'all_experiments_and_diagnostics_verified'
    assert (root / 'analysis_final/REPORT.md').is_file()
    assert (root / 'analysis_final/PROTOCOL.md').is_file()
    source = root / 'final-source.bundle'
    assert not source.exists(), source
    subprocess.run(['git', 'bundle', 'create', str(source), 'HEAD'], check=True)
    files = [source]
    for name in ('analysis_final', 'analysis_training', 'analysis_scale'):
        files.extend(sorted((root / name).glob('*')))
    for name in ('dataset_audit.json', 'ambiguity_probe.json', 'opening_prior.json',
                 'tabular_diagnostics.json', 'recovery_status.json', 'recovery_scale_status.json',
                 'recovery_diagnostics_status.json', 'data_main/recovery_verified.json',
                 'data_expanded/recovery_verified.json', 'local-regression-final.log', 'analysis-targeted-tests.log',
                 'main_provenance_A.json', 'main_provenance_B.json',
                 'remote_completion_A.json', 'remote_completion_B.json',
                 'run_main.py', 'run_expanded.py', 'expanded_stages.py',
                 'run_diagnostics_v2.py', 'verify_remote_completion.py',
                 'source_inputs/SOURCE_INPUTS.json', 'source_inputs/frozen_policy_v4.pt',
                 'junqi_cuda.cpython-312-x86_64-linux-gnu.so', 'native-inputs.sha256', 'runtime-deps.tgz'):
        path = root / name
        assert path.is_file(), path
        files.append(path)
    for prefix in ('training', 'scale', 'diagnostics'):
        for role in 'AB':
            folder = root / f'{prefix}_{role}'
            files.extend(path for path in sorted(folder.rglob('*'))
                         if path.is_file() and path.suffix != '.pt')
    for name in ('data_main', 'data_expanded'):
        files.extend(sorted((root / name).glob('*.pt.json')))
    files = sorted(set(files))
    entries = []
    for path in files:
        with path.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        entries.append((path.relative_to(root).as_posix(), digest, path.stat().st_size))
    readme = (
        '# BeliefNet 公开特征实验交付包\n\n'
        '请先阅读 analysis_final/REPORT.md。图、完整逐种子统计、协议、运行配置、日志、'
        '诊断与源代码 bundle 均包含在本包。\n\n'
        '数据生成用的冻结策略、对应原生扩展及源代码包含在本包。大体积原始数据和 best.pt／last.pt 不重复放入本包，已分别完整保存在本地 '
        + str(root) + ' 下的 data_main、data_expanded、training_A/B、scale_A/B。'
        '对应原始文件哈希保留在各目录 artifacts.sha256 或数据元信息中；'
        '这些原始清单包含未装入 ZIP 的大文件，不能直接作为 ZIP 文件清单验证。\n\n'
        '本包自身以 PACKAGE_SHA256.txt 校验。源代码 bundle 可由 git clone final-source.bundle 恢复；'
        '各阶段的固定提交与运行环境见协议和 launcher_provenance.json。\n'
    )
    manifest = ''.join(f'{digest}  {name}\n' for name, digest, _ in entries)
    readme_digest = hashlib.sha256(readme.encode()).hexdigest()
    manifest += f'{readme_digest}  README.md\n'
    assert not output.exists(), output
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, path.relative_to(root).as_posix())
        archive.writestr('README.md', readme)
        archive.writestr('PACKAGE_SHA256.txt', manifest)
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        for row in manifest.splitlines():
            digest, name = row.split('  ', 1)
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest, name
        count = len(archive.namelist())
    with output.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    result = {'zip': str(output), 'sha256': digest, 'files': count,
              'bytes': output.stat().st_size, 'all_member_hashes_verified': True,
              'raw_data_and_checkpoints_retained_locally': str(root),
              'analysis_commit': analysis['analysis_commit']}
    output.with_suffix('.verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    package(args.root, args.output)
