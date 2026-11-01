Implementation of IEEE 802.11a/g on a AI Engine (XDNA) NPU, based on designs from https://github.com/bastibl/gr-ieee802-11 . 

# Prerequisites
* a Linux machine with XDNA1 NPU
* gnuradio
* mlir-aie https://github.com/Xilinx/mlir-aie with xchesscc compiler
* gr-mlir_aie  https://github.com/philoteer/gr-mlir_aie
* modified gr-foo (from this repo)
* modified gr-ieee802-11 (from this repo)

# Install Direction
* For each GNU Radio module: mkdir build; cd build; cmake ..; make -jn; sudo make install; sudo ldconfig
* For NPU kernels: go into kernel/ directory and run make.

# Directories
* gr-foo-modified: modified gr-foo (original implementation from  https://github.com/bastibl/gr-foo ).
* gr-ieee802-11-modified: modified gr-ieee802-11 (ditto, from https://github.com/bastibl/gr-ieee802-11 )
* base: IEEE 802.11a/g implementation on NPU, with LS equalizer. kernel codes are located under base/kernel.
* AI-model-basic: IEEE 802.11a/g implementation on NPU, with a basic neural network-based demodulator
* AI-model-CNN-Channel-est: ditto, with CNN-based channel estimator (recommended)
