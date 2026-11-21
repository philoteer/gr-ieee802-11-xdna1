# MAC decoder hardware regression

Run on the target NPU using the same environment as `kernel/`:

```sh
make test INPUT_OFFSET=0
make test INPUT_OFFSET=8
make all INPUT_OFFSET=0
```

Use `NPU2=1` on XDNA2. These tests compile the CNN tree's production MAC
decoder and check exact recovered bytes at all eight legacy OFDM rates,
for short through maximum-length frames, with and without sparse decision errors.
Frames spanning kernel calls are included.

`INPUT_OFFSET=8` reproduces the fused graph's non-vector-aligned consumer
buffers. Buffer size does not guarantee vector alignment; the decoder must
use unaligned input loads. The final command restores the input ABI expected
by the GNU Radio test graphs.
