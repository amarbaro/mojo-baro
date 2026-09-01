# shim/tools

`hipblaslt-tune.cpp` — standalone hipBLASLt explorer, no MAX and no Mojo, so it
isolates the vendor library from our runtime. Sweeps every heuristic candidate
against splitK x wgm and reports the best, plus the workspace that config
actually needs.

It is what established that hipBLASLt's defaults leave ~2x on the table on
gfx1100 (2497 -> 5201 GFLOP/s at 512^3 fp32) and that splitK is the workspace
consumer, which is why `amarbaro_ctx` right-sizes rather than reserving a fixed
budget.

```bash
g++ hipblaslt-tune.cpp -o /tmp/hipblaslt-tune -std=c++17 -w \
    -I/opt/rocm/include -L/opt/rocm/lib -lhipblaslt -lamdhip64 -D__HIP_PLATFORM_AMD__
/tmp/hipblaslt-tune 2048     # matrix size; default 512
```

Also useful as the neutral referee when our kernel and the vendor disagree:
it shares no code with `bench/`, so agreement between them is real agreement.
