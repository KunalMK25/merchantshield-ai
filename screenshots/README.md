# Screenshots

The images in this folder are used by the root `README.md`. They are captured from the running
app, never mocked up, and every figure in them is synthetic.

To regenerate them (the app must be running, and Google Chrome installed):

```bash
# terminal 1: API
python -m uvicorn backend.main:app --port 8000
# terminal 2: dashboard dev server
(cd frontend && npm run dev)
# terminal 3: capture (writes the PNGs here)
node scripts/capture_screenshots.mjs http://localhost:5173 screenshots
```

The shots show whatever the running app contains, so run a few retraining cycles first for
meaningful pages (see "Quick start" in the root README): two ordinary days, then one with
`--drift amount_scale=2.5`, then `python -m ml.cases.ingest`.

| File | Page |
|---|---|
| `evaluate.png`, `evaluate-light.png` | Evaluate, a critical demo scenario, dark and light themes |
| `registry.png` | Model registry |
| `retraining.png` | Retraining |
| `drift.png` | Drift & anomalies |
| `alerts.png` | Alerts |
