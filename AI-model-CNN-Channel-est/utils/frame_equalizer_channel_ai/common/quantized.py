from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QuantizedEstimator:
    """Container for the three exported int8 convolution weight tensors.

    Attributes:
        weights: Tuple containing the first hidden, second hidden, and output
            convolution tensors in ``[output, input, tap]`` layout.
    """

    weights: tuple[np.ndarray, np.ndarray, np.ndarray]
    gate_snr_ratio_x16: int = np.iinfo(np.uint64).max
