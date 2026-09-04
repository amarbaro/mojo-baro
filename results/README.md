# Hardware reports

`bench/report.sh` writes its receipts here. Each one records the fp16 WMMA GEMM
kernel against hipBLASLt on one machine, along with the card, the ROCm
toolchain, the commit and the clock state, so a number can always be traced to
the code and the hardware that produced it.

Every published figure in `docs/BASELINE.md` and the README was measured on one
RX 7900 XTX. Nothing here has been shown to transfer to another card. Receipts
from other hardware are how that changes.

To contribute one, see [CONTRIBUTING.md](../CONTRIBUTING.md).

Files are named `report-<gfx>-<card>-<commit>.json`. A receipt with
`"valid": false` is kept rather than discarded — the `problems` array says why
it cannot be quoted, and a failure on hardware we do not own is still a result.
