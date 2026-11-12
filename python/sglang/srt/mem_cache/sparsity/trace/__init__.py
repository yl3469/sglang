"""Real-time indexer-trace collection + offline DP allocation (HiSparse T1).

This subpackage mirrors the ``dsa-offloading`` reference workflow inside
sglang: run real DSv4 inference, capture the per-layer indexer selection
trace (``indexer.topk``) during decode, replay it offline (LRU + Belady miss
curves via :mod:`per_layer_budget_replay`), and produce a per-layer DP
capacity allocation (via :mod:`per_layer_budget_dp`) in the same CSV schema as
the reference ``dp_allocations.csv``.

Pipeline stages (each a standalone module / CLI):

* ``prep_swe_bench``       - SWE-bench -> ``bench_serving`` custom JSONL.
* ``indexer_trace_sink``   - persist ``meta_info["indexer_topk"]`` per response.
* ``record_kv_budget``     - query ``/server_info`` + ``/metrics`` for KV budget.
* ``trace_to_curves``      - decoded topk -> per-layer SelectionTrace -> curves.
* ``run_per_layer_dp``     - DP allocate + reference-schema CSV.

The trace-to-curves and DP stages are pure host-side analysis and are unit
tested on CPU; the serving-side pieces are exercised by the GPU sbatch job.
"""
