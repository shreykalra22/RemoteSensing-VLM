# RemoteSensing-VLM

Research prototype for BigEarthNet-compatible multimodal land-cover analysis.

The production baseline is a pretrained 12-channel BigEarthNet-v2 ResNet-50; it is not a Transformer or VLM. An experimental trainable fusion Transformer is implemented separately and is not yet trained.

## Run locally

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
Set-Location frontend; npm run dev -- --host 127.0.0.1 --port 5173
```

See `docs/` for architecture, API, evaluation status, and limitations.
