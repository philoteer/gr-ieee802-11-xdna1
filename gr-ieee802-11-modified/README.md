Hi!

This an IEEE 802.11 a/g/p transceiver for GNU Radio that is fitted for operation
with Ettus N210s and B210s. Interoperability was tested with many off-the-shelf
WiFi cards and IEEE 802.11p prototypes. The code can also be used in
simulations.

# Table of Contents
1. [Development](#development)

1. [Installation](#installation)

1. [Usage](#usage)

1. [Troubleshooting](#troubleshooting)

1. [Further information](#further-information)

# Development

## Int8-weight channel estimators

The normal and debug frame equalizers expose four additional algorithms in
C++, Python, and GNU Radio Companion:

| Algorithm | Initial channel estimate | Payload channel update |
| --- | --- | --- |
| `CNN_FRAME_INT8` | Frame-split int8 CNN | None |
| `CNN_CHANNEL_INT8` | Channel-split int8 CNN | None |
| `CNN_FRAME_MLP_INT8` | Frame-split int8 CNN | Frame-split int8 MLP |
| `CNN_CHANNEL_MLP_INT8` | Channel-split int8 CNN | Channel-split int8 MLP |

For example, select `ieee802_11.CNN_FRAME_MLP_INT8` as the frame equalizer's
algorithm after rebuilding and installing this module. The enum values are
10 through 13, respectively. Algorithm switching also supports these options.

These use the exported NPU weights and SNR gates from
`AI-model-CNN-Channel-est/kernel/channel_estimator/model/`. The neural-network
MACs are integer operations: signed int8 Q7 weights, Q15 activations, int64
accumulation, floor shifts by 7, and saturation after each layer. CNN
activations saturate to int32; MLP activations saturate to int16, with ReLU on
hidden layers. GNU Radio's complex samples, normalization, residual addition,
SNR measurement, and symbol equalization retain their floating-point interface.
Thus this implements the NPU network arithmetic at a float/Q15 boundary;
whole-receiver bit-exact equivalence to the NPU's integer FFT/input pipeline is
not implied. The NPU exports have different gates from the trained float
exports, so inference may be enabled at different SNRs.

The compact CPU weight headers are checked in. To refresh them after changing
the NPU exports, run `python3 utils/export_channel_int8.py` from this directory
(NumPy required). The script transposes the padded NPU MLP tensors into compact
output-major order and records source hashes; normal builds need no NPU tools.

### Verification and CPU performance

With a Release build in `build/`:

```sh
ctest --test-dir build -R qa_channel_ai --output-on-failure
python3 utils/test_channel_ai_grc.py --build build
python3 utils/test_channel_ai_grc.py --build build --debug-block
build/lib/qa_channel_ai --benchmark by_frame
build/lib/qa_channel_ai --benchmark by_frame --int8
```

Use `by_channel` instead of `by_frame` to benchmark the other weights. The
reference tests load the original NPU exports independently and exercise zero
inputs, large/small amplitudes, normalization boundaries, SNR gate boundaries,
pilot polarity, resets, constellations, and allocation-free inference. Both
block variants passed the GRC symbol corpus at all eight rates, with active and
gated inference. These checks do not establish BER/PER equivalence over a noisy
channel sweep.

Measured on an AMD Ryzen 5 8640HS with GCC 13.3, CMake Release defaults (no
`-march=native`), pinned to CPU 2: median microseconds per active call, taking
the median of three runs, each with 100 warmups and 1,000 measured calls:

| Weights | CNN float | CNN int8 | MLP float | MLP int8 |
| --- | ---: | ---: | ---: | ---: |
| Frame split | 6.27 | 23.80 | 0.61 | 2.71 |
| Channel split | 6.26 | 23.88 | 0.59 | 2.71 |

This portable implementation is about **3.8x slower for the CNN and 4.4–4.6x
slower for the MLP**. Float Eigen/SIMD is efficient for these small models;
integer conversion, wide accumulators, shifts, and saturation add work. This
code does not use a CPU-specific packed int8 dot-product backend. Such a backend
could improve speed, but standard int8-by-int8 kernels would also quantize the
activations more aggressively than this Q15/Q7 contract. The compact int8 tables
use one quarter of the float tables' weight storage (6,864 versus 27,456 bytes
per CNN+MLP pair); both sets are compiled into this module, and activation
buffers are not reduced. For CPU throughput, the measured float versions are
preferable; the integer options provide a useful quantized CPU reference.

Like GNU Radio, this module uses *maint* branches for development.
These branches are supposed to be used with the corresponding GNU Radio
branches. This means: the *maint-3.7* branch is compatible with GNU Radio 3.7,
*maint-3.8* is compatible with GNU Radio 3.8, etc.


# Installation


## Dependencies


### GNU Radio

There are several ways to install GNU Radio. You can use

- [pybombs](http://gnuradio.org/redmine/projects/pybombs/wiki)
- [pre-compiled binaries](http://gnuradio.org/redmine/projects/gnuradio/wiki/BinaryPackages)
- [from source](http://gnuradio.org/redmine/projects/gnuradio/wiki/InstallingGRFromSource)


### gr-foo

I have some non project specific GNU Radio blocks in my gr-foo repo that are
needed. For example the Wireshark connector. You can find these blocks at
[https://github.com/bastibl/gr-foo](https://github.com/bastibl/gr-foo). They are
installed with the typical command sequence:

    git clone https://github.com/bastibl/gr-foo
    cd gr-foo
    mkdir build
    cd build
    cmake ..
    make
    sudo make install
    sudo ldconfig


## Installation of gr-ieee802-11

To actually install the blocks do

    git clone https://github.com/bastibl/gr-ieee802-11
    cd gr-ieee802-11
    mkdir build
    cd build
    cmake ..
    make
    sudo make install
    sudo ldconfig

### Adjust Maximum Shared Memory
Since the transmitter is using the Tagged Stream blocks it has to store a
complete frame in the buffer before processing it. The default maximum shared
memory might not be enough on most Linux systems. It can be increased with

    sudo sysctl -w kernel.shmmax=2147483648


### OFDM PHY

The physical layer is encapsulated in a hierarchical block to allow for a
clearer transceiver structure in GNU Radio Companion. This hierarchical block is
not included in the installation process. You have to open
```/examples/wifi_phy_hier.grc``` with GNU Radio Companion and build it. This
will install the block in ```~/.grc_gnuradio/```.


### Check message port connections

Sometime the connections between the message ports (the gray ones in GNU Radio
Companion) break. Therefore, please open the flow graphs and assert that
everything is connected. It should be pretty obvious how the blocks are supposed
to be wired. Actually this should not happen anymore, so if your ports are still
unconnected please drop me a mail.


### Python OpenGL

If you want to run the receive demo (the one that plots the subcarrier
constellations), please assert that you have python-opengl installed. The nongl
version of the plot does not work for me.


### Run volk_profile

volk_profile is part of GNU Radio. It benchmarks different SIMD implementations
on your PC and creates a configuration file that stores the fastest version of
every function. This can speed up the computation considerably and is required
in order to deal with the high rate of incoming samples.


### Calibrate your daughterboard

If you have a WBX, SBX, or CBX daughterboard you should calibrate it in order to
minimize IQ imbalance and TX DC offsets. See the [application
notes](http://files.ettus.com/manual/page_calibration.html).


# Checking your installation

As a first step I recommend to test the ```wifi_loopback.grc``` flow graph. This
flow graph does not need any hardware and allows you to ensure that the software
part is installed correctly. So open the flow graph and run it. If everything
works as intended you should see some decoded packets on the console.


## Troubleshooting

If GRC complains that it can't find some blocks (other than performance counters
and hierarchical blocks) like

    >>> Error: Block key "ieee802_11_ofdm_mac" not found in Platform - grc(GNU Radio Companion)
    >>> Error: Block key "foo_packet_pad" not found in Platform - grc(GNU Radio Companion)

Most likely you used a different ```CMAKE_INSTALL_PREFIX``` for the module than
for GNU Radio. Therefore, the blocks of the module ended up in a different
directory and GRC can't find them. You have to tell GRC where these blocks are
by creating/adding to your ```~/.gnuradio/config.conf``` something like

    [grc]
    global_blocks_path = /opt/local/share/gnuradio/grc/blocks
    local_blocks_path = /Users/basti/usr/share/gnuradio/grc/blocks

But with the directories that match your installation.


# Usage


## Simulation

The loopback flow graph should give you an idea of how simulations can be
conducted. To ease use, most blocks have debugging and logging capabilities that
can generate traces of the simulation. You can read about the logging feature
and how to use it on the [GNU Radio
Wiki](https://wiki.gnuradio.org/index.php/Logging).


## Unidirectional communication

As first over the air test I recommend to try ```wifi_rx.grc``` and
```wifi_tx.grc```. Just open the flow graphs in GNU Radio companion and execute
them. If it does not work out of the box, try to play around with the gain. If
everything works as intended you should see similar output as in the
```wifi_loopback.grc``` example.


## Ad Hoc Network with WiFi card

- The transceiver is currently connected to a TAP device, i.e. is a virtual
  Ethernet interface. Therefore, we have no WiFi signaling like association
  requests and hence, the transceiver can not "join" an ad hoc network. You have
  to make some small changes to the kernel in order to convince you WiFi card to
  send to this hosts nevertheless.
- The transceiver can not respond to ACKs in time. This is kind of an
  architectural limitation of USRP + GNU Radio since Ethernet and computations
  on a normal CPU introduce some latency. You can set the number of ACK retries
  to zero and handle retransmits on higher layers (-> TCP).
- RTS/CTS is not working for the same reason. You can however just disable this
  mechanism.
- Currently, there is no CSMA/CA mechanism, but this can be implemented on the
  FPGA.


# Troubleshooting

- Please check compile and installation logs. They might contain interesting
  information.
- Did you calibrate your daughterboard?
- Did you run volk_profile?
- Did you try different gain settings?
- Did you close the case of the devices?
- Did you try real-time priority?
- Did you compile GNU Radio and gr-ieee802-11 in release mode?
- If you see warnings that ```blocks_ctrlport_monitor_performance``` is missing
  that means that you installed GNU Radio without control port or performance
  counters. These blocks allow you to monitor the performance of the transceiver
  while it is running, but are not required. You can just delete them from the
  flow graph.
- The message

    You must now use ifconfig to set its IP address. E.g.,
    $ sudo ifconfig tap0 192.168.200.1

is normal and is output by the TUN/Tap Block during startup. The configuration
of the TUN/TAP interface is handled by the scripts in the ```apps``` folder.
- Did you try to tune the RF frequency out of the band of interest (i.e. used
  the LO offset menu of the flow graphs)?
- If 'D's appear, it might be related to your Ethernet card. Assert that you
  made the sysconf changes recommended by Ettus. Did you try to connect you PC
  directly to the USRP without a switch in between?


# Further information

For further information please checkout our project page
[https://www.wime-project.net](https://www.wime-project.net)

For information and an implementation surrounding IEEE 802.11ah, please refer to [gr-ieee802-11ah](https://github.com/irongiant33/gr-ieee802-11ah).
