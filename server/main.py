"""
MProbe backend — Dataset Cleaning & Poison Detection service.

Run with:
    uvicorn server.main:app --reload --port 8000

The static frontend (models/dataset-cleaning.html) calls this API directly
via fetch(). This is the only model in the marketplace wired to a real
backend right now — everything else on the site is still a front-end mock.
"""
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .dataset_io import DatasetError
from .pipeline import OUTPUT_DIR, run_pipeline

SAMPLE_DATASET_PATH = os.path.join(os.path.dirname(__file__), "samples", "sample_dataset.zip")

app = FastAPI(title="MProbe — Dataset Cleaning & Poison Detection")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _run_and_respond(zip_bytes):
    try:
        report, job_id, _output_path = run_pipeline(zip_bytes)
    except DatasetError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "report": report,
        "download_url": f"/v1/models/dataset-cleaning/download/{job_id}",
    }


@app.post("/v1/models/dataset-cleaning/run")
async def run_dataset_cleaning(dataset: UploadFile = File(...)):
    zip_bytes = await dataset.read()
    return _run_and_respond(zip_bytes)


@app.post("/v1/models/dataset-cleaning/run-sample")
async def run_dataset_cleaning_sample():
    if not os.path.exists(SAMPLE_DATASET_PATH):
        raise HTTPException(status_code=404, detail="Sample dataset not found")
    with open(SAMPLE_DATASET_PATH, "rb") as f:
        zip_bytes = f.read()
    return _run_and_respond(zip_bytes)


@app.get("/v1/models/dataset-cleaning/download/{job_id}")
async def download_cleaned_dataset(job_id: str):
    path = os.path.join(OUTPUT_DIR, f"{job_id}_cleaned.zip")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Result expired or not found")
    return FileResponse(path, filename="cleaned_dataset.zip", media_type="application/zip")
