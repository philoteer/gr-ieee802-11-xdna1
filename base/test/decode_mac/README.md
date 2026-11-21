# MAC decoder hardware regression

Run on the target NPU using the same environment as `base/kernel`:

```sh
make test INPUT_OFFSET=0
make test INPUT_OFFSET=8
make all INPUT_OFFSET=0
```

Use `NPU2=1` on XDNA2. The tests generate valid scrambled, convolutionally
encoded, punctured, and interleaved DATA symbols for all eight legacy OFDM
rates. They check exact recovered bytes for short and maximum-length frames,
with and without sparse decision errors, including frames spanning kernel calls.
The final command restores the input ABI expected by the GNU Radio test graphs.

`INPUT_OFFSET=8` deliberately misaligns decisions relative to the FIFO base.
The fused XDNA1 graph can place its consumer buffers at addresses such as
`0x72b88` and `0x76308`: a 1024-byte buffer size does **not** guarantee vector
alignment. The decoder must use unaligned loads on this external input;
aligned standalone FIFOs previously hid the bug. Scratch arrays remain aligned.

After changing the shared header, rebuild the production graph with
`make -C ../../kernel all`.
