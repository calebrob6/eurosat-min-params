"""Export complete locked results without changing historical blog artifacts."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import re

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.data import TIFF_BAND_NAMES

from .protocol import sha256, write_json

INDEX_DEFINITIONS = {
    'ndvi': '(B08 - B04) / (B08 + B04 + epsilon)',
    'ndwi': '(B03 - B08) / (B03 + B08 + epsilon)',
    'ndbi': 'legacy alias: (B12 - B08) / (B12 + B08 + epsilon)',
    'ndmi': 'legacy alias: (B08 - B12) / (B08 + B12 + epsilon)',
    'nbr': 'legacy alias: (B08 - B8A) / (B08 + B8A + epsilon)',
    'bsi': 'legacy alias: ((B12 + B04) - (B08 + B02)) / ((B12 + B04) + (B08 + B02) + epsilon)',
}


def physical_definition(name: str) -> str:
    for index, definition in INDEX_DEFINITIONS.items():
        if index in name:
            return definition
    bands = re.findall(r'(?:^|_)b(\d+)', name)
    if bands:
        return ';'.join(TIFF_BAND_NAMES[int(index)] for index in bands)
    if 'pan' in name:
        return 'mean of all 13 TIFF channels; B10 fixed to zero in b10_zeroed variant'
    for band in TIFF_BAND_NAMES:
        if name.endswith('_' + band):
            return band
    raise ValueError(f'unknown physical feature definition: {name}')


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f'cannot export an empty result table: {path}')
    columns = sorted({key for row in rows for key in row})
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def family_summary(features: list[dict]) -> list[dict]:
    groups = {}
    for row in features:
        key = tuple(row[field] for field in ('model', 'variant', 'representation', 'family'))
        groups.setdefault(key, []).append(row['r2'])
    result = []
    for key, scores in sorted(groups.items()):
        values = np.asarray([value for value in scores if value is not None])
        result.append({
            **dict(zip(('model', 'variant', 'representation', 'family'), key, strict=True)),
            'features': len(scores), 'undefined_features': len(scores) - len(values),
            'macro_r2': float(values.mean()) if len(values) else None,
            'median_r2': float(np.median(values)) if len(values) else None,
            'q25_r2': float(np.quantile(values, .25)) if len(values) else None,
            'q75_r2': float(np.quantile(values, .75)) if len(values) else None,
        })
    return result


def figures(destination: Path, results: dict, models: list[str]) -> None:
    reps = ('frontier33', 'pool377', 'imagestats52')
    scores = np.full((len(models), len(reps)), np.nan)
    for row in results['evaluations']:
        if (row['kind'] == 'ridge' and row['context'] == 'ordinary'
                and row['variant'] == 'original' and row['direction'] == 'h_to_z'):
            scores[models.index(row['model']), reps.index(row['representation'])] = 100 * row['native_r2']
    fig, ax = plt.subplots(figsize=(7, 4))
    image = ax.imshow(scores, aspect='auto', cmap='viridis')
    ax.set_xticks(range(len(reps)), reps)
    ax.set_yticks(range(len(models)), models)
    for i in range(len(models)):
        for j in range(len(reps)):
            ax.text(j, i, f'{scores[i, j]:.1f}', ha='center', va='center', color='white')
    fig.colorbar(image, ax=ax, label='Held-out embedding variance explained (%)')
    fig.tight_layout()
    fig.savefig(destination / 'explained_variance.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for offset, rep in zip((-.12, .12), reps[:2], strict=True):
        rows = [next(row for row in results['comparisons']
                     if row['model'] == model and row['representation'] == rep and row['primary'])
                for model in models]
        centers = np.asarray([row['accuracy_delta'] for row in rows]) * 100
        bounds = np.asarray([[row['accuracy_ci_low'], row['accuracy_ci_high']] for row in rows]) * 100
        positions = np.arange(len(models)) + offset
        # Draw intervals directly: a percentile interval need not contain its point estimate.
        ax.vlines(positions, bounds[:, 0], bounds[:, 1])
        ax.plot(positions, centers, 'o', label=rep)
    ax.axhline(0, color='black', linewidth=.7)
    ax.set_xticks(range(len(models)), models, rotation=15, ha='right')
    ax.set_ylabel('Accuracy change (percentage points)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination / 'accuracy_complementarity.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(models), figsize=(3.2 * len(models), 3), squeeze=False)
    for ax, model in zip(axes[0], models, strict=True):
        for rep in reps[:2]:
            rows = [row for row in results['ranks']
                    if row['model'] == model and row['variant'] == 'original' and row['representation'] == rep]
            rows.sort(key=lambda row: row['rank'])
            ax.plot([row['rank'] for row in rows], [row['prediction_reconstruction_r2'] for row in rows], label=rep)
        if rows:
            ax.plot([row['rank'] for row in rows], [row['oracle_reconstruction_r2'] for row in rows],
                    linestyle='--', color='gray', label='Target PCA reference')
        ax.set_xscale('log')
        ax.set_title(model)
        ax.set_xlabel('Train-PCA components')
        ax.set_ylabel('Full-target R-squared')
    axes[0, -1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(destination / 'pca_reconstruction.png', dpi=160)
    plt.close(fig)


def format_score(value: float | None, multiplier: float = 1) -> str:
    return 'undefined' if value is None else f'{value * multiplier:.2f}'


def export(output: Path, destination: Path) -> None:
    from .run import locked_jobs

    jobs = locked_jobs(output)
    results_path = output / 'results.json'
    results = json.loads(results_path.read_text())
    protocol = json.loads((output / 'protocol.json').read_text())
    if results['protocol_sha256'] != sha256(output / 'protocol.json'):
        raise ValueError('result protocol differs from locked selection')
    if {row['id'] for row in results['evaluations']} != {job['id'] for job in jobs}:
        raise ValueError('results omit or add locked jobs')
    expected_primary = 2 * len(protocol['models'])
    if sum(row['primary'] for row in results['comparisons']) != expected_primary:
        raise ValueError('wrong primary comparison count')
    identity = {'protocol_sha256': results['protocol_sha256'], 'results_sha256': sha256(results_path)}
    if destination.exists() and any(destination.iterdir()):
        old = destination / 'export.json'
        if not old.exists() or json.loads(old.read_text()) != identity:
            raise ValueError(f'export destination contains another run: {destination}')
    destination.mkdir(parents=True, exist_ok=True)
    feature_rows = [{**row, 'physical_definition': physical_definition(row['feature'])}
                    for row in results['features']]
    families = family_summary(feature_rows)
    selection_rows = []
    for job in jobs:
        for candidate in job['sweep']:
            selection_rows.append({
                **{key: job[key] for key in ('id', 'kind', 'model', 'variant', 'representation', 'context')},
                'direction': job.get('direction', ''),
                **{key: value for key, value in candidate.items() if not isinstance(value, (dict, list))},
            })
    tables = {
        'selection.csv': selection_rows,
        'decoding.csv': [row for row in results['evaluations'] if row['kind'] == 'ridge'],
        'classification.csv': [row for row in results['evaluations'] if row['kind'] == 'probe'],
        'complementarity.csv': results['comparisons'],
        'per_class.csv': results['per_class'],
        'features.csv': feature_rows,
        'families.csv': families,
        'geometry.csv': results['geometry'],
        'pca_reference.csv': results['ranks'],
        'paired_reconstruction.csv': results['paired_reconstruction'],
    }
    for name, rows in tables.items():
        write_csv(destination / name, rows)
    inputs = json.loads((output / 'input_manifest.json').read_text())
    inputs['handcrafted']['raw_source'] = 'data/EuroSAT'
    write_json(destination / 'provenance.json', {'protocol': protocol, 'inputs': inputs})
    figures(destination, results, protocol['models'])
    lines = [
        '# EuroSAT representation overlap results',
        '',
        'New controlled analyses of frozen embeddings and raw handcrafted features. These do not replace the article scoreboard. H33 and H377 are not strictly nested: H33 includes one region-shape measurement outside H377.',
        '',
        '| Backbone | H33 explains Z (%) | H377 explains Z (%) | H33 accuracy gain (pp) | H377 accuracy gain (pp) |',
        '|---|---:|---:|---:|---:|',
    ]
    for model in protocol['models']:
        values = []
        for rep in ('frontier33', 'pool377'):
            row = next(row for row in results['evaluations'] if row['kind'] == 'ridge'
                       and row['model'] == model and row['variant'] == 'original'
                       and row['representation'] == rep and row['context'] == 'ordinary'
                       and row['direction'] == 'h_to_z')
            values.append(format_score(row['native_r2'], 100))
        for rep in ('frontier33', 'pool377'):
            row = next(row for row in results['comparisons']
                       if row['model'] == model and row['representation'] == rep and row['primary'])
            values.append(format_score(row['accuracy_delta'], 100))
        lines.append('| ' + ' | '.join([model, *values]) + ' |')
    lines.extend([
        '',
        'The explanation percentages are native-coordinate held-out multivariate R-squared, not fractions of information or learned parameters. Reverse-decoding measurements and family summaries are in `features.csv` and `families.csv`; class-conditioned and shuffled decoding controls are in `decoding.csv`.',
        '',
        'All validation candidate scores and scalar selection diagnostics are exported in `selection.csv`. Boundary choices describe the tested finite grids, not globally optimal regularization.',
        '',
        'Accuracy gains compare a separately regularized, block-scaled concatenation with the controlled backbone-only probe. `complementarity.csv` includes paired intervals, corrected/error counts, exact McNemar p-values, Holm adjustment for the primary comparisons, shuffled and redundant-projection controls, and B10-disabled OlmoEarth input ablations. Log loss is secondary.',
        '',
        f'Intervals use {protocol["bootstrap"]} paired image bootstrap draws and are conditional on the fitted models. They do not measure feature-discovery, training-sample, or geographic uncertainty. EuroSAT was already extensively used in the earlier research; this is not an untouched external confirmation.',
        '',
        'The PCA reconstruction reference sees the target embedding and is not a guaranteed held-out ceiling or an H-to-Z predictor. CKA describes geometry, not directional containment. Neither low linear decodability nor a probe gain proves information-theoretic absence from a backbone.',
        '',
    ])
    (destination / 'README.md').write_text('\n'.join(lines))
    write_json(destination / 'export.json', identity)
