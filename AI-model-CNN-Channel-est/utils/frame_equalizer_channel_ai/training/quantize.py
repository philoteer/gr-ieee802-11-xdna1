import numpy as np

from ..common.model import ChannelEstimator
from ..common.quantized import QuantizedEstimator


def quantize_model(model: ChannelEstimator,
                   gate_snr_ratio_x16: int = np.iinfo(np.uint64).max) -> QuantizedEstimator:
    weights = tuple(np.clip(np.rint(layer.weight.detach().numpy() * 128.0),
                            -128, 127).astype(np.int8) for layer in model.layers)
    return QuantizedEstimator(weights, gate_snr_ratio_x16)  # type: ignore[arg-type]
