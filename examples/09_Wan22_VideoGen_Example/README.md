# WAN 2.2 text-to-video (wan-2.2-t2v-a14b)

Client configs for a video-generation server such as `trtllm-serve` with
`POST /v1/videos/generations`. How T2V maps onto MLPerf Endpoints (point set,
metrics, checker, open WG items) is in
[`docs/t2v/README.md`](../../docs/t2v/README.md).

## Endpoints points

Each point is one run. `CONCURRENCY` sets the point; `N_SAMPLES` is at least 50
(§6.4) and large enough to meet the §6.2 minimum duration (600 s at concurrency
1-32, 1200 s above). The accuracy run covers all 248 prompts once. `ENDPOINT`
and `SERVED_MODEL` default to `http://localhost:8000` and `wan22`.

- `endpoints_point_wan22.yaml`: performance, then VBench accuracy on the same
  endpoint. Use it for every point that carries an accuracy result.
- `endpoints_point_wan22_perf.yaml`: performance only, for a submitter's-choice
  point.

```bash
CFG=examples/09_Wan22_VideoGen_Example/endpoints_point_wan22.yaml
for c in 1 4 8; do
  CONCURRENCY=$c N_SAMPLES=248 inference-endpoint benchmark from-config -c "$CFG"
done
CONCURRENCY=33 N_SAMPLES=496 inference-endpoint benchmark from-config -c "$CFG"
```

## Server

- **Fixed latent.** Load the reference `fixed_latent.pt` on the server (for
  trtllm-serve, `TLLM_VISUAL_GEN_FIXED_LATENT_PATH`) and declare its sha256 as
  `fixed_latent_sha256` in each `point.yaml`. The client refuses a `latent_path`
  column.
- **Reference settings.** The request defaults are the reference settings. A run
  with `submission_ref` refuses prompt rows that override them.

## After each run

- **Check `n_samples_failed == 0`.** A failed clip produces no frames, and the
  checker fails a video point that has any.
- **Use `max_issue_duration_ms` only as a runaway guard.** A binding cap stops
  the point short of its sample count and breaks §6.4.
- **Declare warmup in `point.yaml` (§6.3.3).** The server warms up at startup;
  list that under `initialization_steps`. A client warmup (`warmup.enabled`)
  needs `warmup.salt`, because the accuracy and performance prompts are the same
  file (§6.3.1).
- **Keep `model_params.name` set to the served model name.** When it is unset,
  the client fills it from `submission_ref.model` and sends the slug to the
  server.
- **Check the upload** with the cohort checker:
  `python -m inference_endpoint.validation <cohort> <upload>`.
