"""Native CPU/CUDA, zero-learned-parameter EuroSAT feature extraction.

Use ``EuroSATFeatures('33')`` before an ordinary trainable ``torch.nn.Linear``.
Inputs are raw, unnormalized ``(N, 13, 64, 64)`` TIFF-order tensors; outputs are
float32 tensors on the input device. Feature extraction is not differentiable.
See ``EuroSATFeatures`` for the historical channel naming and numeric contract.
"""

from ._extractor import EuroSATFeatures
from ._schema import TIFF_BAND_NAMES

__all__ = ['EuroSATFeatures', 'TIFF_BAND_NAMES']
