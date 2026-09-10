"""Native tensor arithmetic shared by the fixed feature families."""
import math

import torch
from torch.nn import functional as F

EPS = 1e-6


def pool2(x):
    n, c, h, w = x.shape
    return x.reshape(n, c, h // 2, 2, w // 2, 2).mean((3, 5))


def mean_std(x):
    mean = x.mean(-1)
    # NumPy uses a two-pass population variance, not Welford's algorithm.
    z = x - mean.unsqueeze(-1)
    return mean, (z * z).mean(-1).sqrt()


def percentile(x, q):
    """NumPy 2.5 linear percentile, including scalar-q weak promotion."""
    scalar = isinstance(q, (int, float))
    ordered = x.sort(dim=-1).values
    pos = (torch.tensor([q] if scalar else q, dtype=torch.float64, device=x.device) / 100) * (x.shape[-1] - 1)
    lo, hi = pos.floor().long(), pos.ceil().long()
    a, b = ordered[..., lo], ordered[..., hi]
    diff = b - a
    weight = pos - lo
    if scalar:
        # NumPy treats Python scalar q as weak: both the product and sum round
        # to float32, not just the final result. This matters for strict > edges.
        return torch.where(weight >= 0.5, b - diff * (1 - weight).to(x.dtype),
                           a + diff * weight.to(x.dtype))[..., 0]
    diff = diff.double()
    return torch.where(weight >= 0.5, b.double() - diff * (1 - weight), a.double() + diff * weight)


def median(x):
    ordered = x.sort(dim=-1).values
    size = x.shape[-1]
    if size % 2:
        return ordered[..., size // 2]
    return (ordered[..., size // 2 - 1] + ordered[..., size // 2]) / 2


def gradients(x):
    return x[..., :-1, 1:] - x[..., :-1, :-1], x[..., 1:, :-1] - x[..., :-1, :-1]


def magnitude(x):
    gx, gy = gradients(x)
    power = gx * gx + gy * gy
    # The CPU float32 sqrt kernel can differ by one ULP from NumPy's correctly
    # rounded sqrt. Rounding a double result prevents false threshold crossings.
    if power.device.type == "cpu":
        return power.double().sqrt().to(power.dtype)
    return power.sqrt()


def index_maps(x):
    nir, red, green, blue, sw1, sw2 = (x[:, i] for i in (7, 3, 2, 1, 11, 12))
    pairs = ((nir, red), (green, nir), (sw1, nir), (nir, sw1), (nir, sw2))
    maps = [(a - b) / (a + b + EPS) for a, b in pairs]
    maps.append(((sw1 + red) - (nir + blue)) / ((sw1 + red) + (nir + blue) + EPS))
    return torch.stack(maps, 1)


def pan(x):
    # NumPy sums a non-contiguous band axis sequentially. Matching that order
    # matters for strict threshold/LBP comparisons on tied pixels.
    total = x[:, 0].clone()
    for i in range(1, 13):
        total = total + x[:, i]
    # CUDA's scalar float32 division may multiply by an approximate reciprocal;
    # that creates false edges even in integer-valued linear ramps.
    return (total.double() / 13).float()


def coherence(x):
    gx, gy = gradients(x)
    sxx, syy, sxy = ((a * b).mean((-2, -1)) for a, b in ((gx, gx), (gy, gy), (gx, gy)))
    return ((sxx - syy).square() + 4 * sxy * sxy).sqrt() / (sxx + syy + EPS)


def orientation_histogram(x, bins):
    gx, gy = gradients(x)
    mag = (gx * gx + gy * gy).sqrt().flatten(-2)
    angle = torch.atan2(gy, gx)
    # NumPy's float32 atan2 puts the positive diagonal one ULP below pi/4.
    # CUDA and scalar CPU libm round it up instead. Preserve the historical
    # bin for tied integer gradients, independently of batch/kernel dispatch.
    angle = torch.where((gx == gy) & (gx > 0), 0.7853981256484985, angle)
    angle = torch.remainder(angle, math.pi)
    idx = (angle / (math.pi / bins)).long().clamp(max=bins - 1).flatten(-2)
    # The reference accumulates each bin in float32 and normalizes in float64.
    hist = torch.stack([torch.where(idx == b, mag, 0).sum(-1) for b in range(bins)], -1).double()
    return hist / (hist.sum(-1, keepdim=True) + EPS)


def orientation_entropy(x):
    hist = orientation_histogram(x, 8)
    return -(hist * (hist + EPS).log()).sum(-1).float()


def uniform_filter3(x):
    """SciPy's separable 3x3 filter: half-sample symmetric ('reflect') edges.

    For radius one these edges equal replicate padding, not reflection_pad.
    Each axis accumulates in double and rounds its result to the input dtype.
    """
    y = torch.cat((x[..., :1, :], x, x[..., -1:, :]), -2).double()
    y = ((y[..., :-2, :] + y[..., 1:-1, :]) + y[..., 2:, :]) / 3
    y = y.to(x.dtype)
    z = torch.cat((y[..., :1], y, y[..., -1:]), -1).double()
    return (((z[..., :-2] + z[..., 1:-1]) + z[..., 2:]) / 3).to(x.dtype)


def harris(x):
    gy, gx = torch.gradient(x, dim=(-2, -1), edge_order=1)
    jxx, jyy, jxy = (uniform_filter3(a * b) for a, b in ((gx, gx), (gy, gy), (gx, gy)))
    trace = jxx + jyy
    response = (jxx * jyy - jxy * jxy) - 0.05 * trace * trace
    trace, response = trace.flatten(-2), response.flatten(-2)
    active = trace > median(trace).unsqueeze(-1)
    frac = ((response > 0) & active).sum(-1).double() / (active.sum(-1).double() + EPS)
    mag = response.clamp_min(0).sqrt().sum(-1) / (trace.sum(-1) + EPS)
    return torch.stack((frac.float(), mag), -1).flatten(1)


def hough_geometry():
    size, ntheta = 63, 60
    diag = math.hypot(size, size)
    nrho = math.ceil(diag) + 1
    y, x = torch.meshgrid(torch.arange(size, dtype=torch.float64),
                          torch.arange(size, dtype=torch.float64), indexing="ij")
    theta = torch.arange(ntheta, dtype=torch.float64) * (math.pi / ntheta)
    rho = theta.cos()[:, None] * x.flatten() + theta.sin()[:, None] * y.flatten()
    bins = ((rho + diag) / 2).floor().long().clamp(0, nrho - 1)
    return bins, nrho, diag


def hough(x, bins, nrho, diag):
    mag = magnitude(x).flatten(-2)
    threshold = percentile(mag, 85)
    edge = (mag > threshold.unsqueeze(-1)).float()
    flat = edge.reshape(-1, edge.shape[-1])
    # Chunk angles to bound working memory even for large image batches.
    peaks = []
    for chunk in bins.split(10):
        acc = torch.zeros((len(flat), len(chunk), nrho), dtype=torch.float32, device=x.device)
        acc.scatter_add_(2, chunk.unsqueeze(0).expand(len(flat), -1, -1),
                         flat.unsqueeze(1).expand(-1, len(chunk), -1))
        peaks.append(acc.amax(-1))
    per_angle = torch.cat(peaks, -1)
    peak = per_angle.amax(-1).reshape(edge.shape[:-1])
    top3 = per_angle.topk(3, dim=-1).values.sum(-1).reshape(edge.shape[:-1])
    count = edge.sum(-1) + EPS
    return torch.stack((peak / count, (peak.double() / diag).float(), top3 / count), -1).flatten(1)


def component_edges(height=64, width=64):
    nodes = torch.arange(1, height * width + 1).reshape(height, width)
    left, right = [], []
    for dy, dx in ((0, 1), (1, -1), (1, 0), (1, 1)):
        ya, yb = slice(0, height - dy), slice(dy, height)
        xa = slice(max(0, -dx), min(width, width - dx))
        xb = slice(max(0, dx), min(width, width + dx))
        left.append(nodes[ya, xa].flatten())
        right.append(nodes[yb, xb].flatten())
    return torch.stack((torch.cat(left), torch.cat(right)))


def component_sizes(mask, edges=None):
    """Sizes indexed by root label for batched 8-connected binary masks.

    Parallel union-by-min with pointer jumping runs until convergence, with no
    image-diameter iteration cap (which would fail on winding/hollow regions).
    Only convergence scalars synchronize; all labels stay on the input device.
    """
    height, width = mask.shape[-2:]
    flat = mask.reshape(-1, height * width)
    if edges is None:
        edges = component_edges(height, width).to(mask.device)
    valid = F.pad(flat, (1, 0), value=False)
    a, b = (side.unsqueeze(0).expand(len(flat), -1) for side in edges)
    active = valid.gather(1, a) & valid.gather(1, b)
    a, b = torch.where(active, a, 0), torch.where(active, b, 0)
    parents = torch.arange(height * width + 1, device=mask.device).expand(len(flat), -1).clone()
    while True:
        old = parents.clone()
        ra, rb = parents.gather(1, a), parents.gather(1, b)
        parents.scatter_reduce_(1, torch.maximum(ra, rb), torch.minimum(ra, rb),
                                reduce="amin", include_self=True)
        parents = parents.gather(1, parents)
        if torch.equal(old, parents):
            break
    sizes = torch.zeros_like(parents)
    sizes.scatter_add_(1, parents[:, 1:], flat.long())
    return sizes.reshape(*mask.shape[:-2], height * width + 1)


def blobs(x, edges):
    flat = x.flatten(-2)
    mask = x > median(flat)[..., None, None]
    sizes = component_sizes(mask, edges)
    count = (sizes > 0).sum(-1)
    largest = sizes.amax(-1).double() / flat.shape[-1]
    average = sizes.sum(-1).double() / count.clamp_min(1) / flat.shape[-1]
    return torch.stack((largest, count.double().log1p(), average), -1).float().flatten(1)


def lbp(x):
    h, w = x.shape[-2:]
    center = x[..., 1:-1, 1:-1]
    offsets = ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))
    bits = torch.stack([x[..., 1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx] > center
                        for dy, dx in offsets], 0)
    transitions = (bits != bits.roll(1, 0)).sum(0)
    riu = torch.where(transitions <= 2, bits.sum(0), 9).flatten(-2)
    hist = torch.stack([(riu == i).sum(-1) for i in range(10)], -1).double() / riu.shape[-1]
    p = hist.clamp_min(1e-12)
    entropy = -(p * p.log()).sum(-1)
    return torch.stack((entropy, hist[..., :9].sum(-1)), -1).float().flatten(1)


def tail_anisotropy(ndvi):
    flat = ndvi.flatten(-2)
    q25 = percentile(flat, [25])[..., 0].float()
    weights = (q25[..., None] - flat).clamp_min(0)
    # np.linspace calculates in float64 before casting to float32.
    coords = torch.linspace(-1, 1, 64, dtype=torch.float64, device=flat.device).float()
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    xx, yy = xx.flatten(), yy.flatten()
    mass = weights.sum(-1) + EPS
    mean_x, mean_y = ((weights * coord).sum(-1) / mass for coord in (xx, yy))
    dx, dy = xx - mean_x[..., None], yy - mean_y[..., None]
    sxx, syy, sxy = ((weights * a * b).sum(-1) / mass for a, b in ((dx, dx), (dy, dy), (dx, dy)))
    return ((sxx - syy).square() + 4 * sxy * sxy).sqrt() / (sxx + syy + EPS)
