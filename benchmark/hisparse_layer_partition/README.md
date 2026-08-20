# HiSparse per-layer DP partition: online profiling + benchmark

End-to-end recipe used on 4x GB300 with DeepSeek-V4-Flash-0731 (FP8). The
partition is solved ONLINE from a short profiling run of the served model —
no offline trace infrastructure.

## Pipeline

1. **Capture** (eager server, capped selection capture + memory report):

   ```
   bash launch_server.sh capture      # --disable-cuda-graph, hisparse-config:
                                      #   selection_capture {path, max_steps_per_layer}
                                      #   memory_report_path
   bash run_bench.sh 127.0.0.1 capture capture   # long prompts fill the cap
   ```

   Full indexer logits are deliberately NOT captured (they are
   sequence-sized); the top-k selection stream is all the LRU miss-curve
   replay needs, and `max_steps_per_layer` bounds memory/disk hard.

2. **Solve** (LRU replay over the size grid + exact min-plus DP; budget
   derived from the per-rank free-memory reports, overridable):

   ```
   python -m sglang.srt.mem_cache.sparsity.solve_dp \
     --capture cap.rank0.pt --reports 'memreport.rank*.json' \
     --out dp.json --target-mean 3072
   ```

3. **Benchmark** (equal effective mean capacity):

   ```
   bash launch_server.sh baseline   # uniform at dp.json:uniform_mean
   bash launch_server.sh dp         # device_buffer_size=max(sizes) + profile
   bash run_bench.sh 127.0.0.1 <tag> sweep   # 24k-in rate sweep
   bash run_bench.sh 127.0.0.1 <tag> long    # 64k-in concurrency sweep
   python summarize_bench.py                 # CSV + curve plot
   ```

## Result (2026-08-20, 4x GB300, TP4, top_k=512, physical buffer 4096 slots)

Solved profile at mean 3072: sizes 1024..4096, predicted −15.0% host-load
misses vs uniform-3072 on the captured 24k-token traffic. Served
throughput/latency, however, was **indistinguishable from uniform at 24k
contexts** (median TPOT 45.5 vs 46.4 ms at the contended point; all rates
within noise), and slightly WORSE at 64k contexts (TPOT 48.3 vs 35.5 ms at
concurrency 8) where the 24k-solved profile transfers out of distribution
and its smallest layers (1024 slots) become miss hotspots.

Interpretation: on GB300 the C2C host-load path is fast enough that a
~15% miss-count reduction is not on the decode critical path at these
operating points. The partition mechanism is correct and free when off
(bit-identical default); the win case needs either slower host links,
larger host:device ratios, or profiles solved on traffic matching the
serving distribution.
