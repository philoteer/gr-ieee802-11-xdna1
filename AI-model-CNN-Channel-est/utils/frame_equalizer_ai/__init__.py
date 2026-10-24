"""CPU training and export tools for the frame-equalizer neural demapper.

This package contains a deterministic PyTorch pipeline that fits one
bias-free MLP per 802.11 modulation (BPSK, QPSK, 16-QAM, 64-QAM) for the
architecture selected with ``--model``, quantizes the weights to the
AIE-facing integer contract, and exports them as the C++ header consumed by
the learned equalizer engine.  The entry point is ``cli.py``; see
``README.md`` for the end-to-end workflow.

Modules:
    cli: ``train``/``export``/``validate`` subcommands and orchestration.
    constellations: GNU Radio's exact constellation class-index ordering.
    dataset: compiles/runs the data.grc generator and loads its output.
    model: the MLP architectures, their registry, and CPU training helpers.
    quantize: symmetric int8 quantization and integer reference inference.
    exporter: NPZ/JSON/C++ header packing of the quantized models.
"""
