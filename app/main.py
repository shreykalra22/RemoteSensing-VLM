"""HTTP boundary for the RemoteSensing-VLM baseline application."""

import json
import logging
import tempfile
import zipfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.services.bigearthnet_service import DEVICE
from app.services.scene_pipeline_service import run_scene_pipeline

LOGGER = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parents[1]
MAX_ARCHIVE_BYTES = 200 * 1024 * 1024
MAX_EXTRACTED_BYTES = 500 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 200

app = FastAPI(title="RemoteSensing-VLM", description="BigEarthNet baseline scene-analysis API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict:
    return {"name": "RemoteSensing-VLM", "status": "running", "model": "BigEarthNet-v2 ResNet-50"}


@app.get("/health")
def health() -> dict:
    return {"status": "healthy", "model": "BigEarthNet-v2 ResNet-50", "device": str(DEVICE), "pipeline": "baseline"}


def _evaluation_summary(filename: str, name: str, state: str) -> dict:
    path = BASE_DIR / "results" / filename
    if not path.is_file():
        return {"name": name, "state": "unavailable", "reason": f"No stored result file: results/{filename}"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {"name": name, "state": state, "metrics": {key: payload.get(key) for key in ("macro_f1", "micro_f1", "macro_average_precision", "micro_average_precision")}, "support": payload.get("support"), "labels": payload.get("labels"), "evaluation": payload.get("evaluation"), "dataset_summary": payload.get("dataset_summary")}
    except (OSError, json.JSONDecodeError) as exc:
        return {"name": name, "state": "unavailable", "reason": f"Stored evaluation cannot be read: {exc}"}


@app.get("/api/evaluations")
def evaluations() -> dict:
    """Expose persisted evaluations only; this endpoint never invents metrics."""
    return {"baseline": _evaluation_summary("baseline-phase3-test.json", "Pretrained BigEarthNet-v2 ResNet-50", "evaluated"), "fusion": _evaluation_summary("fusion-phase3-final-test.json", "S1/S2 fusion Transformer (bounded 3-epoch experiment)", "trained_limited"), "text_modality": {"state": "unavailable", "reason": "No legitimate leakage-free text source has been configured."}}


def _default_directories() -> tuple[Path, Path]:
    root = BASE_DIR / "reben-training-scripts" / "scripts" / "data"
    return root / "S1", root / "S2"


@app.post("/predict")
def predict_default() -> dict:
    s1_dir, s2_dir = _default_directories()
    # The legacy sample directories are retained for local inspection only.
    # Pair validation below intentionally rejects unrelated acquisitions.
    return _run_pipeline(s1_dir, s2_dir)


def _safe_extract(archive_path: Path, destination: Path) -> None:
    if not zipfile.is_zipfile(archive_path):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid ZIP archive.")
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        if len(members) > MAX_ARCHIVE_MEMBERS or sum(item.file_size for item in members) > MAX_EXTRACTED_BYTES:
            raise HTTPException(status_code=400, detail="ZIP archive exceeds the allowed extracted size or file count.")
        for item in members:
            target = (destination / item.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise HTTPException(status_code=400, detail="ZIP archive contains an unsafe file path.")
        archive.extractall(destination)


async def _save_upload(upload: UploadFile, target: Path, name: str) -> None:
    if not upload.filename or not upload.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail=f"{name} must be supplied as a .zip archive.")
    written = 0
    with target.open("wb") as buffer:
        while chunk := await upload.read(1024 * 1024):
            written += len(chunk)
            if written > MAX_ARCHIVE_BYTES:
                raise HTTPException(status_code=413, detail=f"{name} exceeds the {MAX_ARCHIVE_BYTES // 1024 // 1024} MB upload limit.")
            buffer.write(chunk)


def _run_pipeline(s1_dir: Path, s2_dir: Path) -> dict:
    try:
        return run_scene_pipeline(s1_dir, s2_dir)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # Do not disclose filesystem paths or tracebacks to clients.
        LOGGER.exception("Scene pipeline failed")
        raise HTTPException(status_code=500, detail="Scene analysis failed. Check the server logs for details.") from exc


@app.post("/predict/upload")
async def predict_upload(s1_zip: UploadFile = File(...), s2_zip: UploadFile = File(...)) -> dict:
    with tempfile.TemporaryDirectory(prefix="remote_sensing_") as temporary_directory:
        workspace = Path(temporary_directory)
        s1_archive, s2_archive = workspace / "s1.zip", workspace / "s2.zip"
        s1_dir, s2_dir = workspace / "s1", workspace / "s2"
        s1_dir.mkdir()
        s2_dir.mkdir()
        try:
            await _save_upload(s1_zip, s1_archive, "Sentinel-1 upload")
            await _save_upload(s2_zip, s2_archive, "Sentinel-2 upload")
            _safe_extract(s1_archive, s1_dir)
            _safe_extract(s2_archive, s2_dir)
            return _run_pipeline(s1_dir, s2_dir)
        finally:
            await s1_zip.close()
            await s2_zip.close()
