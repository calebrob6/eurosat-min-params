#!/usr/bin/env python
"""Run the article experiments in their separate CPU and CUDA environments."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
STEPS = ('local', 'backbones', 'overlap', 'baselines', 'figures')


def digest(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def products(name: str, output: Path) -> list[Path]:
    files = {
        'local': ['local/models.csv', 'local/environment.json', 'local/frontier_fractions.csv',
                  'local/imagestats_fractions.csv', 'local/features_306.csv', 'local/per_class.csv'],
        'backbones-comparison': ['backbones/scoreboard_comparison.csv', 'backbones/fraction_comparison.csv'],
        'overlap-prepare': ['features/features.json'],
        'overlap-select': ['overlap/selection-decoding.json', 'overlap/selection-probes.json'],
        'overlap-lock': ['overlap/locked.json'],
        'overlap-evaluate': ['overlap/results.json'],
        'overlap-report': ['overlap-results/export.json'],
        'overlap-comparison': ['overlap-comparison.csv'],
    }
    directories = {
        'local': ['local'],
        'overlap-prepare': ['features'],
        'overlap-select': ['overlap/selected'],
        'overlap-evaluate': ['overlap/predictions', 'overlap/evaluated'],
        'overlap-report': ['overlap-results'],
        'baselines': ['baselines'],
        'figures': ['figure-statistics'],
        'article-tables': ['article-tables'],
        'tiny-cnn': ['tiny-cnn'],
        'mosaiks': ['mosaiks'],
    }
    result = {output / filename for filename in files.get(name, [])}
    if name in ('backbones-scoreboard', 'backbones-curves', 'overlap-embeddings'):
        if name == 'backbones-scoreboard':
            models, datasets = ('earthloc', 'moco', 'resnet18', 'vit_base'), ('eurosat',)
        elif name == 'backbones-curves':
            models = ('resnet50', 'convnext_tiny', 'dofa_base', 'dofa_large',
                      'olmoearth_nano', 'olmoearth_small', 'olmoearth_base')
            datasets = ('eurosat', 'eurosat-spatial')
        else:
            models = ('resnet50', 'convnext_tiny', 'dofa_large', 'olmoearth_nano', 'olmoearth_base')
            datasets = ('eurosat',)
        for model in models:
            for dataset in datasets:
                stem = f'{model}_{dataset}'
                result.update((output / f'backbones/{stem}.json', output / f'embeddings/{stem}.npz'))
                if name != 'overlap-embeddings' and dataset == 'eurosat':
                    result.add(output / f'backbones/scoreboard/{stem}.csv')
                if name == 'backbones-curves':
                    result.add(output / f'backbones/{stem}.csv')
    for folder in directories.get(name, []):
        paths = {path for path in (output / folder).rglob('*') if path.is_file()}
        if not paths:
            raise ValueError(f'{name}: no results were written under {output / folder}')
        result.update(paths)
    if not result or any(not path.is_file() for path in result):
        raise ValueError(f'{name}: a required output is missing')
    return sorted(result)


def source_hashes() -> dict[str, str]:
    paths = {*ROOT.glob('*.py'), *ROOT.glob('requirements*.txt')}
    for folder in ('src', 'experiments'):
        paths.update((ROOT / folder).rglob('*.py'))
    return {path.relative_to(ROOT).as_posix(): digest(path) for path in sorted(paths)}


def reference_hashes() -> dict[str, str]:
    paths = set((ROOT / 'submissions').glob('*/model.npz'))
    for folder in (
        'results', 'experiments/imported', 'experiments/torchgeo_bench_eurosat/reproduced',
        'experiments/representation_overlap/results', 'experiments/article_baselines/measured',
        'experiments/article_baselines/archived',
    ):
        paths.update(path for path in (ROOT / folder).rglob('*')
                     if path.is_file() and path.suffix in ('.csv', '.json', '.txt'))
    paths.add(ROOT / 'experiments/article_baselines/archive_manifest.csv')
    return {path.relative_to(ROOT).as_posix(): digest(path) for path in sorted(paths)}


def commands(args) -> list[tuple[str, list[str]]]:
    cpu, gpu = str(args.cpu_python), str(args.benchmark_python)
    output = args.output
    download = ['--download'] if args.download else []
    check = ['--check'] if args.check else []
    backbones = output / 'backbones'
    embeddings = output / 'embeddings'
    overlap = output / 'overlap'
    features = output / 'features'
    jobs = []
    if 'local' in args.steps:
        jobs.append(('local', [cpu, 'reproduce.py', '--fractions', '--output', str(output / 'local'), *download, *check]))
    if 'backbones' in args.steps:
        common = [gpu, 'experiments/torchgeo_bench_eurosat/run.py', '--output', str(backbones),
                  '--cache', str(embeddings), '--device', args.device, *download]
        jobs.append(('backbones-scoreboard', [*common, '--models', 'earthloc', 'moco', 'resnet18', 'vit_base']))
        jobs.append(('backbones-curves', [
            *common, '--models', 'resnet50', 'convnext_tiny', 'dofa_base', 'dofa_large',
            'olmoearth_nano', 'olmoearth_small', 'olmoearth_base',
            '--datasets', 'eurosat', 'eurosat-spatial', '--fractions',
        ]))
        jobs.append(('backbones-comparison', [cpu, 'experiments/torchgeo_bench_eurosat/compare.py',
                                             '--results-dir', str(backbones)]))
    if 'overlap' in args.steps:
        if 'backbones' not in args.steps:
            jobs.append(('overlap-embeddings', [
                gpu, 'experiments/torchgeo_bench_eurosat/run.py', '--extract-only',
                '--models', 'resnet50', 'convnext_tiny', 'dofa_large', 'olmoearth_nano', 'olmoearth_base',
                '--output', str(backbones), '--cache', str(embeddings), '--device', args.device, *download,
            ]))
        common = [gpu, '-m', 'experiments.representation_overlap.run']
        options = ['--output', str(overlap), '--cache', str(features),
                   '--embeddings', str(embeddings), '--metadata', str(backbones), '--device', args.device]
        jobs.append(('overlap-prepare', [*common, 'prepare', *options, *download]))
        for stage in ('select', 'lock', 'evaluate'):
            jobs.append((f'overlap-{stage}', [*common, stage, *options]))
        jobs.append(('overlap-report', [*common, 'report', *options, '--export', str(output / 'overlap-results')]))
        jobs.append(('overlap-comparison', [
            cpu, 'compare_overlap.py', '--results', str(output / 'overlap-results'),
            '--output', str(output / 'overlap-comparison.csv'), *check,
        ]))
    if 'baselines' in args.steps:
        jobs.append(('baselines', [cpu, '-m', 'experiments.article_baselines.run',
                                   '--output', str(output / 'baselines'), *download, *check]))
    if 'figures' in args.steps:
        jobs.append(('figures', [cpu, 'experiments/figure_statistics.py',
                                '--output', str(output / 'figure-statistics'), *download, *check]))
    if args.exploratory:
        jobs.append(('tiny-cnn', [
            gpu, 'experiments/article_baselines/other.py', '--method', 'tiny-cnn',
            '--gpu', args.device.split(':')[-1], '--epochs', '60',
            '--output-dir', str(output / 'tiny-cnn'),
        ]))
        jobs.append(('mosaiks', [
            gpu, 'experiments/article_baselines/other.py', '--method', 'mosaiks',
            '--device', args.device, '--output-dir', str(output / 'mosaiks'),
        ]))
    jobs.append(('article-tables', [sys.executable, 'export_blog_results.py', '--output', str(output / 'article-tables')]))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', choices=STEPS, nargs='+', default=list(STEPS))
    parser.add_argument('--cpu-python', type=Path, default=ROOT / '.venv-reproduce/bin/python')
    parser.add_argument('--benchmark-python', type=Path, default=ROOT / 'output/benchmark-env/bin/python')
    parser.add_argument('--output', type=Path, default=ROOT / 'output/reproduce-all')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--exploratory', action='store_true', help='also run the expensive CNN and MOSAIKS configurations as new experiments')
    parser.add_argument('--check', action='store_true', help='fail if local or overlap results differ from reference values')
    parser.add_argument('--dry-run', action='store_true', help='print commands without running them')
    args = parser.parse_args()
    if len(set(args.steps)) != len(args.steps):
        parser.error('--steps must not contain duplicates')
    if args.exploratory and not (args.device.startswith('cuda:') and args.device[5:].isdigit()):
        parser.error('--exploratory requires an explicit --device cuda:N')
    args.output = args.output.resolve()
    if not args.output.is_relative_to(ROOT / 'output'):
        parser.error('place --output inside the ignored output/ directory in this repository')
    args.cpu_python = args.cpu_python.absolute()
    args.benchmark_python = args.benchmark_python.absolute()
    jobs = commands(args)
    for name, command in jobs:
        print(f'{name}: {shlex.join(command)}', flush=True)
    if args.dry_run:
        return
    for _, command in jobs:
        if not Path(command[0]).is_file():
            raise FileNotFoundError(f'{command[0]}: set up the environment in REPRODUCIBILITY.md first')
    args.output.mkdir(parents=True, exist_ok=True)
    specification = {'commands': jobs, 'source_sha256': source_hashes(), 'reference_sha256': reference_hashes()}
    specification = json.loads(json.dumps(specification))
    manifest = args.output / 'run.json'
    if manifest.exists():
        state = json.loads(manifest.read_text())
        if state['specification'] != specification:
            raise ValueError('reproduction commands changed; use a new output directory')
    else:
        state = {'specification': specification, 'completed': {}, 'running': None}
    def save() -> None:
        temporary = manifest.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(manifest)
    for name, command in jobs:
        if name in state['completed']:
            for relative, expected in state['completed'][name].items():
                if digest(args.output / relative) != expected:
                    raise ValueError(f'{name}: a completed result changed: {relative}')
            print(f'{name}: already completed', flush=True)
            continue
        state['running'] = name
        save()
        subprocess.run(command, cwd=ROOT, check=True)
        state['completed'][name] = {
            path.relative_to(args.output).as_posix(): digest(path)
            for path in products(name, args.output)
        }
        state['running'] = None
        save()
    print(f'Results saved to {args.output}')


if __name__ == '__main__':
    main()
