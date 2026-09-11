"""Frozen feature order; names retain the historical numeric band identifiers."""

TIFF_BAND_NAMES = (
    'B01',
    'B02',
    'B03',
    'B04',
    'B05',
    'B06',
    'B07',
    'B08',
    'B09',
    'B10',
    'B11',
    'B12',
    'B8A',
)
INDICES = ('ndvi', 'ndwi', 'ndbi', 'ndmi', 'nbr', 'bsi')
CHANNELS = ('pan', 'ndvi', 'ndbi')
PAIRS = ((3, 7), (2, 7), (1, 7), (11, 7), (3, 11), (2, 3), (7, 12), (11, 12))
POOL_INDICES = (
    325,
    297,
    316,
    294,
    144,
    94,
    145,
    348,
    206,
    195,
    298,
    75,
    37,
    317,
    287,
    86,
    291,
    323,
    95,
    17,
    310,
    29,
    63,
    197,
    68,
    329,
    314,
    289,
    65,
    32,
    365,
    2,
)
GROUPS = (
    (
        'spectral_statistics',
        tuple(
            f'{stat}_b{i}'
            for stat in ('mean', 'std', 'p10', 'p25', 'p50', 'p75', 'p90')
            for i in range(13)
        ),
    ),
    (
        'multiscale_gradients',
        tuple(
            f'g{s}{stat}_b{i}'
            for s in range(3)
            for stat in ('mean', 'std')
            for i in range(13)
        ),
    ),
    ('coherence', tuple(f'coh{s}_b{i}' for s in range(2) for i in range(13))),
    ('orientation_entropy', tuple(f'oent_b{i}' for i in range(13))),
    (
        'orientation_histogram',
        tuple(f'hog0_b{i}_o{b}' for i in range(13) for b in range(4)),
    ),
    ('spectral_peaks', tuple(f'fftpk_b{i}' for i in range(13))),
    ('cross_band', tuple(f'xcorr_b{a}_b{b}' for a, b in PAIRS)),
    (
        'index_texture',
        tuple(
            f'ix{stat}_{name}'
            for stat in ('std', 'gm', 'gs', 'spr')
            for name in INDICES
        ),
    ),
    (
        'hough_lines',
        tuple(f'line{stat}_{name}' for name in CHANNELS for stat in ('pf', 'pl', 't3')),
    ),
    (
        'harris_corners',
        tuple(f'corn2{stat}_{name}' for name in CHANNELS for stat in ('frac', 'mag')),
    ),
    (
        'lbp',
        tuple(f'lbp2{stat}_{name}' for name in CHANNELS for stat in ('ent', 'uni')),
    ),
    (
        'blobs',
        tuple(
            f'blob2{stat}_{name}' for name in CHANNELS for stat in ('lrg', 'nc', 'msz')
        ),
    ),
    ('spectral_slope', tuple(f'sslope2_{name}' for name in CHANNELS)),
    ('index_coherence', tuple(f'ixcoh_{name}' for name in INDICES)),
    (
        'index_texture_scale2',
        tuple(f'ix2{stat}_{name}' for stat in ('std', 'gm') for name in INDICES),
    ),
    ('cross_band_correlation', tuple(f'xcorr_b{a}_b{b}' for a, b in PAIRS)),
    ('orientation_entropy_scale2', tuple(f'oent2_b{i}' for i in range(13))),
)
POOL_NAMES = tuple(name for _, names in GROUPS for name in names)
REGION_SHAPE_NAMES = tuple(
    f'tail_{stat}_{tail}_{channel}'
    for channel in CHANNELS
    for tail in ('low', 'high')
    for stat in ('aniso', 'spread')
)
EXTENDED_POOL_NAMES = POOL_NAMES + REGION_SHAPE_NAMES
FRONTIER_NAMES = tuple(POOL_NAMES[i] for i in POOL_INDICES) + ('tail_aniso_low_ndvi',)
STATS_NAMES = tuple(
    f'{stat}_{band}'
    for band in TIFF_BAND_NAMES
    for stat in ('mean', 'std', 'min', 'max')
)
