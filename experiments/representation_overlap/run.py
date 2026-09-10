"""Prepare, select, lock, and evaluate the prespecified representation study."""
from __future__ import annotations

import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
from dataclasses import asdict, fields
import json
import logging
from pathlib import Path

import numpy as np

from . import data, decoding, probes
from .protocol import (
    ALPHAS, BETAS, BOOTSTRAP, CS, MODELS, NULL_SEEDS, PROBE_MAX_ITER, PROBE_TOL, RANKS,
    REPRESENTATIONS, ROOT, SIZES, json_value, require_same, save_npz,
    sha256, specification, write_json,
)

logger = logging.getLogger(__name__)
SPLIT_SEEDS = {'train': 0, 'val': 1, 'test': 2}


def variants(model: str) -> tuple[str, ...]:
    return ('original', 'b10_zeroed') if model.startswith('olmoearth') else ('original',)


def representation_keys(variant: str, pilot: bool) -> tuple[str, ...]:
    if pilot:
        return ('frontier33',)
    return tuple(REPRESENTATIONS) if variant == 'original' else ('frontier33', 'pool377')


def class_means(x: np.ndarray, labels: np.ndarray) -> np.ndarray:
    if not np.array_equal(np.unique(labels), np.arange(10)):
        raise ValueError('class-conditioned analysis requires every EuroSAT class in train')
    return np.stack([x[labels == label].mean(axis=0, dtype=np.float64) for label in range(10)])


def shuffled(x: np.ndarray, split: str, seed: int, labels: np.ndarray | None = None) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([seed, SPLIT_SEEDS[split]]))
    indices = np.arange(len(x))
    if labels is None:
        indices = rng.permutation(indices)
    else:
        for label in np.unique(labels):
            positions = np.flatnonzero(labels == label)
            indices[positions] = rng.permutation(positions)
    return x[indices]


def projection(z: np.ndarray, width: int) -> np.ndarray:
    weights = np.random.default_rng(0).normal(size=(z.shape[1], width)) / np.sqrt(z.shape[1])
    return z @ weights


def job_id(kind: str, model: str, variant: str = '', rep: str = '', context: str = '', direction: str = '') -> str:
    return '-'.join(value for value in (kind, model, variant, rep, context, direction) if value)


def expected_job_ids(models: list[str], pilot: bool = False) -> set[str]:
    identifiers = {
        job_id('probe', 'handcrafted', variant, rep)
        for variant in ('original', 'b10_zeroed')
        for rep in representation_keys(variant, pilot)
    }
    ridge_contexts = ('ordinary',) if pilot else ('ordinary', 'within_class', *(f'shuffle{seed}' for seed in NULL_SEEDS))
    for model in models:
        identifiers.add(job_id('probe', model, context='standalone'))
        for variant in variants(model):
            for rep in representation_keys(variant, pilot):
                for context in ridge_contexts:
                    for direction in ('h_to_z', 'z_to_h'):
                        identifiers.add(job_id('ridge', model, variant, rep, context, direction))
                probe_contexts = ('combined',) if pilot or variant != 'original' else ('combined', 'shuffled', 'projection')
                for context in probe_contexts:
                    identifiers.add(job_id('probe', model, variant, rep, context))
    return identifiers


def save_job(output: Path, job: dict, fitted, sweep: list[dict], **extra) -> dict:
    path = output / 'selected' / f'{job["id"]}.npz'
    save_npz(path, **asdict(fitted), **extra)
    result = {**job, 'artifact': str(path.relative_to(output)), 'artifact_sha256': sha256(path),
              'sweep': sweep}
    write_json(output / 'selected' / f'{job["id"]}.json', result)
    return result


def existing_job(output: Path, identifier: str) -> dict | None:
    path = output / 'selected' / f'{identifier}.json'
    if not path.exists():
        return None
    job = json.loads(path.read_text())
    if job['id'] != identifier or sha256(output / job['artifact']) != job['artifact_sha256']:
        raise ValueError(f'corrupt selected job: {identifier}')
    return job


def load_fitted(output: Path, job: dict):
    cls = decoding.RidgeMap if job['kind'] == 'ridge' else probes.Probe
    with np.load(output / job['artifact'], allow_pickle=False) as handle:
        values = {field.name: handle[field.name] for field in fields(cls)}
        values = {key: value.item() if value.ndim == 0 else value for key, value in values.items()}
        extras = {key: handle[key] for key in handle.files if key not in values}
    return cls(**values), extras


def initialize(args) -> dict:
    feature_manifest = data.load_feature_manifest(args.cache)
    metadata = {
        model: data.backbone_metadata(model, args.embeddings, args.metadata)
        for model in args.models
    }
    spec = {
        **specification(args.models, args.device),
        'pilot': args.pilot,
        'feature_manifest_sha256': sha256(args.cache / 'features.json'),
        'backbones': {model: {'embedding_sha256': item['embedding_sha256'],
                              'backbone_state_sha256': item['backbone_state_sha256']}
                      for model, item in metadata.items()},
    }
    path = args.output / 'protocol.json'
    if path.exists():
        require_same(path, spec)
    else:
        write_json(path, spec)
        write_json(args.output / 'input_manifest.json', {
            'handcrafted': feature_manifest, 'backbones': metadata,
        })
    if (args.output / 'locked.json').exists():
        raise ValueError('selection is locked; use a new output directory for another experiment')
    return json_value(spec)


def select_decoding(args, handcrafted: dict) -> list[dict]:
    jobs = []
    contexts = ('ordinary',) if args.pilot else ('ordinary', 'within_class', *(f'shuffle{seed}' for seed in NULL_SEEDS))
    for model in args.models:
        backbone = data.load_backbone(model, args.embeddings, ('train', 'val'))
        z = {split: backbone[split]['features'] for split in backbone}
        labels = {split: backbone[split]['labels'] for split in backbone}
        z_means = class_means(z['train'], labels['train'])
        z_centered = {split: z[split] - z_means[labels[split]] for split in z}
        z_sources = {}
        for variant in variants(model):
            for rep in representation_keys(variant, args.pilot):
                h = {split: handcrafted[variant][split][rep] for split in ('train', 'val')}
                h_means = class_means(h['train'], labels['train'])
                for context in contexts:
                    current_h, current_z = h, z
                    if context == 'within_class':
                        current_h = {split: h[split] - h_means[labels[split]] for split in h}
                        current_z = z_centered
                    elif context.startswith('shuffle'):
                        seed = int(context.removeprefix('shuffle'))
                        current_h = {split: shuffled(h[split], split, seed) for split in h}
                    for direction in ('h_to_z', 'z_to_h'):
                        identifier = job_id('ridge', model, variant, rep, context, direction)
                        old = existing_job(args.output, identifier)
                        if old is not None:
                            jobs.append(old)
                            continue
                        source, target = (current_h, current_z) if direction == 'h_to_z' else (current_z, current_h)
                        factor = None
                        if direction == 'z_to_h':
                            key = 'within_class' if context == 'within_class' else 'ordinary'
                            if key not in z_sources:
                                z_sources[key] = decoding.prepare_ridge_source(source['train'])
                            factor = z_sources[key]
                        fitted, sweep = decoding.select_ridge(
                            source['train'], target['train'], source['val'], target['val'], ALPHAS,
                            scoring='native' if direction == 'h_to_z' else 'macro',
                            source=factor,
                        )
                        job = {
                            'id': identifier, 'kind': 'ridge', 'model': model, 'variant': variant,
                            'representation': rep, 'context': context, 'direction': direction,
                            'selected_alpha': fitted.alpha,
                        }
                        jobs.append(save_job(
                            args.output, job, fitted, sweep, h_class_means=h_means,
                            z_class_means=z_means, target_train_mean=target['train'].mean(0, dtype=np.float64),
                        ))
                        logger.info('selected %s alpha=%g', identifier, fitted.alpha)
    return jobs


def select_probes(args, handcrafted: dict) -> list[dict]:
    jobs = []
    original = handcrafted['original']
    ytr, yva = original['train']['labels'], original['val']['labels']
    for variant in ('original', 'b10_zeroed'):
        for rep in representation_keys(variant, args.pilot):
            identifier = job_id('probe', 'handcrafted', variant, rep)
            old = existing_job(args.output, identifier)
            if old is not None:
                jobs.append(old)
                continue
            fitted, sweep = probes.select_probe(
                handcrafted[variant]['train'][rep], handcrafted[variant]['val'][rep],
                ytr, yva, CS, (0.0,), primary_standardize=True, device=args.device,
                max_iter=PROBE_MAX_ITER, tol=PROBE_TOL,
            )
            jobs.append(save_job(args.output, {
                'id': identifier, 'kind': 'probe', 'model': 'handcrafted',
                'variant': variant, 'representation': rep, 'context': 'standalone',
            }, fitted, sweep))
            logger.info('selected %s C=%g', identifier, fitted.C)
    for model in args.models:
        backbone = data.load_backbone(model, args.embeddings, ('train', 'val'))
        ztr, zva = backbone['train']['features'], backbone['val']['features']
        identifier = job_id('probe', model, context='standalone')
        old = existing_job(args.output, identifier)
        if old is None:
            fitted, sweep = probes.select_probe(
                ztr, zva, ytr, yva, CS, (0.0,), device=args.device,
                max_iter=PROBE_MAX_ITER, tol=PROBE_TOL,
            )
            old = save_job(args.output, {
                'id': identifier, 'kind': 'probe', 'model': model,
                'variant': 'original', 'representation': '', 'context': 'standalone',
            }, fitted, sweep)
        jobs.append(old)
        for variant in variants(model):
            for rep in representation_keys(variant, args.pilot):
                htr, hva = handcrafted[variant]['train'][rep], handcrafted[variant]['val'][rep]
                contexts = ('combined',) if args.pilot or variant != 'original' else ('combined', 'shuffled', 'projection')
                for context in contexts:
                    identifier = job_id('probe', model, variant, rep, context)
                    old = existing_job(args.output, identifier)
                    if old is not None:
                        jobs.append(old)
                        continue
                    extra_tr, extra_va = htr, hva
                    if context == 'shuffled':
                        extra_tr, extra_va = shuffled(htr, 'train', 0), shuffled(hva, 'val', 0)
                    elif context == 'projection':
                        extra_tr, extra_va = projection(ztr, htr.shape[1]), projection(zva, htr.shape[1])
                    fitted, sweep = probes.select_probe(
                        ztr, zva, ytr, yva, CS, BETAS,
                        extra_train=extra_tr, extra_val=extra_va, device=args.device,
                        max_iter=PROBE_MAX_ITER, tol=PROBE_TOL,
                    )
                    jobs.append(save_job(args.output, {
                        'id': identifier, 'kind': 'probe', 'model': model,
                        'variant': variant, 'representation': rep, 'context': context,
                    }, fitted, sweep))
                    logger.info('selected %s C=%g beta=%g', identifier, fitted.C, fitted.beta)
    return jobs


def select(args) -> None:
    initialize(args)
    handcrafted = {variant: data.load_handcrafted(args.cache, variant, ('train', 'val'))
                   for variant in ('original', 'b10_zeroed')}
    components = ('decoding', 'probes') if args.component == 'all' else (args.component,)
    for component in components:
        jobs = select_decoding(args, handcrafted) if component == 'decoding' else select_probes(args, handcrafted)
        write_json(args.output / f'selection-{component}.json', {
            'protocol_sha256': sha256(args.output / 'protocol.json'),
            'jobs': jobs, 'complete': True,
        })


def locked_jobs(output: Path) -> list[dict]:
    lock_path = output / 'locked.json'
    if not lock_path.exists():
        raise ValueError('test evaluation requires a locked selection; run the lock stage first')
    lock = json.loads(lock_path.read_text())
    for name, digest in lock['files'].items():
        if sha256(output / name) != digest:
            raise ValueError(f'locked artifact changed: {name}')
    return lock['jobs']


def lock_selection(args) -> None:
    spec = json.loads((args.output / 'protocol.json').read_text())
    if spec['pilot']:
        raise ValueError('a pilot cannot be locked or evaluated on test')
    if (args.output / 'locked.json').exists():
        locked_jobs(args.output)
        logger.info('selection already locked')
        return
    initialize(args)
    jobs = []
    paths = ['protocol.json', 'input_manifest.json', 'selection-decoding.json', 'selection-probes.json']
    for component in ('decoding', 'probes'):
        selection = json.loads((args.output / f'selection-{component}.json').read_text())
        if not selection['complete'] or selection['protocol_sha256'] != sha256(args.output / 'protocol.json'):
            raise ValueError(f'{component}: incomplete or mismatched selection')
        jobs.extend(selection['jobs'])
    if len({job['id'] for job in jobs}) != len(jobs):
        raise ValueError('duplicate selected jobs')
    if {job['id'] for job in jobs} != expected_job_ids(spec['models']):
        raise ValueError('selection does not cover exactly the prespecified analysis conditions')
    paths.extend(job['artifact'] for job in jobs)
    write_json(args.output / 'locked.json', {
        'jobs': jobs, 'files': {name: sha256(args.output / name) for name in paths},
    })
    logger.info('locked %d jobs without evaluating test', len(jobs))


def evaluate_ridge(job: dict, fitted, extras: dict, h: np.ndarray, z: np.ndarray, labels: np.ndarray) -> tuple:
    if job['context'] == 'within_class':
        h = h - extras['h_class_means'][labels]
        z = z - extras['z_class_means'][labels]
    elif job['context'].startswith('shuffle'):
        h = shuffled(h, 'test', int(job['context'].removeprefix('shuffle')))
    source, target = (h, z) if job['direction'] == 'h_to_z' else (z, h)
    prediction = fitted.predict(source)
    scores, per_feature = decoding.score_predictions(target, prediction)
    mean_scores, _ = decoding.score_predictions(
        target, np.broadcast_to(extras['target_train_mean'], target.shape),
    )
    scores['train_mean_baseline_r2'] = mean_scores['native_r2']
    scores['source_train_constant_count'] = int(fitted.source_constant.sum())
    scores['source_constant_varies_on_test'] = int(
        (np.ptp(source[:, fitted.source_constant], axis=0) != 0).sum()
    )
    if job['context'] in ('ordinary', 'within_class'):
        scores.update(decoding.bootstrap_r2(target, prediction, n_boot=BOOTSTRAP, seed=0))
    return scores, per_feature, prediction


def evaluate(args) -> None:
    jobs = locked_jobs(args.output)
    spec = json.loads((args.output / 'protocol.json').read_text())
    if spec['pilot']:
        raise ValueError('pilot test access is forbidden')
    current = specification(spec['models'], spec['device'])
    for key, value in current.items():
        if json_value(value) != spec[key]:
            raise ValueError(f'protocol changed before test evaluation: {key}')
    data.load_feature_manifest(args.cache)
    if sha256(args.cache / 'features.json') != spec['feature_manifest_sha256']:
        raise ValueError('handcrafted input manifest changed after selection')
    for model in spec['models']:
        meta = data.backbone_metadata(model, args.embeddings, args.metadata)
        if meta['embedding_sha256'] != spec['backbones'][model]['embedding_sha256']:
            raise ValueError('backbone embeddings changed after selection')
    handcrafted = {variant: data.load_handcrafted(args.cache, variant, ('train', 'test'))
                   for variant in ('original', 'b10_zeroed')}
    labels = handcrafted['original']['test']['labels']
    evaluations = []
    feature_rows, geometry_rows, rank_rows = [], [], []
    feature_schema = json.loads((args.cache / 'features.json').read_text())['schema']
    for model in ('handcrafted', *spec['models']):
        backbone = None if model == 'handcrafted' else data.load_backbone(model, args.embeddings, ('train', 'test'))
        pca = None if backbone is None else decoding.fit_target_pca(backbone['train']['features'])
        for job in (item for item in jobs if item['model'] == model):
            fitted, extras = load_fitted(args.output, job)
            rep, variant = job['representation'], job['variant']
            h = handcrafted[variant]['test'][rep] if rep else None
            result = {key: job[key] for key in ('id', 'kind', 'model', 'variant', 'representation', 'context')}
            path = args.output / 'predictions' / f'{job["id"]}.npz'
            if job['kind'] == 'ridge':
                scores, per_feature, prediction = evaluate_ridge(
                    job, fitted, extras, h, backbone['test']['features'], labels,
                )
                result.update(direction=job['direction'], selected_alpha=fitted.alpha, **scores)
                if job['context'] == 'ordinary':
                    save_npz(path, prediction=prediction)
                    if job['direction'] == 'z_to_h':
                        schema = feature_schema[rep]
                        for index, score in enumerate(per_feature):
                            feature_rows.append({
                                'model': model, 'variant': variant, 'representation': rep,
                                'position': index, 'feature': schema['names'][index],
                                'family': schema['families'][index], 'r2': score,
                            })
                    else:
                        ranks = sorted(set(rank for rank in RANKS if rank <= min(backbone['train']['features'].shape)))
                        for row in pca.diagnostics(backbone['test']['features'], prediction, ranks):
                            rank_rows.append({'model': model, 'variant': variant, 'representation': rep, **row})
                        htr = handcrafted[variant]['train'][rep].astype(np.float64)
                        hmean, hscale = htr.mean(0), htr.std(0)
                        hscale[hscale == 0] = 1
                        hs = (h - hmean) / hscale
                        z = backbone['test']['features']
                        for seed in (-1, *NULL_SEEDS):
                            hx = hs if seed == -1 else shuffled(hs, 'test', seed)
                            geometry_rows.append({
                                'model': model, 'variant': variant, 'representation': rep,
                                'condition': 'paired' if seed == -1 else 'shuffled',
                                'seed': seed, 'cka': decoding.linear_cka(hx, z),
                            })
                            if seed != -1:
                                geometry_rows.append({
                                    'model': model, 'variant': variant, 'representation': rep,
                                    'condition': 'within_class_shuffled', 'seed': seed,
                                    'cka': decoding.linear_cka(shuffled(hs, 'test', seed, labels), z),
                                })
            else:
                primary = h if model == 'handcrafted' else backbone['test']['features']
                extra = None if job['context'] == 'standalone' else h
                if job['context'] == 'shuffled':
                    extra = shuffled(h, 'test', 0)
                elif job['context'] == 'projection':
                    extra = projection(primary, h.shape[1])
                probabilities = fitted.predict_proba(primary, extra)
                result.update(selected_C=fitted.C, selected_beta=fitted.beta,
                              **probes.classification_metrics(labels, probabilities))
                save_npz(path, probabilities=probabilities)
            evaluations.append(result)
            write_json(args.output / 'evaluated' / f'{job["id"]}.json', result)
            logger.info('evaluated %s', job['id'])
    comparisons = []
    per_class = []
    for job in jobs:
        if job['kind'] != 'probe' or job['context'] == 'standalone':
            continue
        baseline_id = job_id('probe', job['model'], context='standalone')
        with np.load(args.output / 'predictions' / f'{baseline_id}.npz') as handle:
            baseline = handle['probabilities']
        with np.load(args.output / 'predictions' / f'{job["id"]}.npz') as handle:
            combined = handle['probabilities']
        comparison = {key: job[key] for key in ('model', 'variant', 'representation', 'context')}
        comparison.update(probes.paired_metrics(labels, baseline, combined, n_boot=BOOTSTRAP, seed=0))
        comparison['primary'] = (job['variant'] == 'original' and job['context'] == 'combined'
                                 and job['representation'] != 'imagestats52')
        comparisons.append(comparison)
        for label, name in enumerate(data.CLASSES):
            selected = labels == label
            per_class.append({
                **{key: job[key] for key in ('model', 'variant', 'representation', 'context')},
                'class': name, 'samples': int(selected.sum()),
                'backbone_correct': int((baseline[selected].argmax(1) == label).sum()),
                'combined_correct': int((combined[selected].argmax(1) == label).sum()),
            })
    primary = [row for row in comparisons if row['primary']]
    adjusted = probes.holm_adjust([row['mcnemar_pvalue'] for row in primary])
    for row, pvalue in zip(primary, adjusted, strict=True):
        row['mcnemar_holm_pvalue'] = pvalue
    paired_reconstruction = []
    for model in spec['models']:
        ztest = data.load_backbone(model, args.embeddings, ('test',))['test']['features']
        for variant in variants(model):
            predictions = []
            for rep in ('frontier33', 'pool377'):
                identifier = job_id('ridge', model, variant, rep, 'ordinary', 'h_to_z')
                with np.load(args.output / 'predictions' / f'{identifier}.npz') as handle:
                    predictions.append(handle['prediction'])
            paired_reconstruction.append({
                'model': model, 'variant': variant,
                **decoding.paired_r2(ztest, predictions[0], predictions[1], n_boot=BOOTSTRAP, seed=0),
            })
    write_json(args.output / 'results.json', {
        'protocol_sha256': sha256(args.output / 'protocol.json'),
        'evaluations': evaluations, 'comparisons': comparisons, 'per_class': per_class,
        'features': feature_rows, 'geometry': geometry_rows, 'ranks': rank_rows,
        'paired_reconstruction': paired_reconstruction,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('all', 'prepare', 'select', 'lock', 'evaluate', 'report'))
    parser.add_argument('--cache', type=Path, default=ROOT / 'output/representation-features')
    parser.add_argument('--embeddings', type=Path, default=ROOT / 'output/benchmark-embeddings')
    parser.add_argument('--metadata', type=Path, default=ROOT / 'output/backbones')
    parser.add_argument('--output', type=Path, default=ROOT / 'output/representation-study')
    parser.add_argument('--export', type=Path, default=ROOT / 'output/representation-results')
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--component', choices=('all', 'decoding', 'probes'), default='all')
    parser.add_argument('--pilot', action='store_true', help='frontier-only validation pilot; cannot access test')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    if args.stage == 'all':
        if args.pilot or args.component != 'all':
            parser.error('the all stage requires a complete, non-pilot study')
        data.prepare(args.cache, args.batch_size, download=args.download)
        select(args)
        lock_selection(args)
        evaluate(args)
        from .report import export
        export(args.output, args.export)
    elif args.stage == 'prepare':
        data.prepare(args.cache, args.batch_size, download=args.download)
    elif args.stage == 'select':
        select(args)
    elif args.stage == 'lock':
        lock_selection(args)
    elif args.stage == 'evaluate':
        evaluate(args)
    else:
        from .report import export
        export(args.output, args.export)


if __name__ == '__main__':
    main()
