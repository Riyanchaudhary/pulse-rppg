# PULSE

PULSE is a local research workstation for measuring facial remote photoplethysmography (rPPG) signals from video. It separates signal quality from deepfake classification. The default installation has no trained classifier and never emits an authentic/manipulated verdict.

## Quick start

Backend (Python 3.11+ recommended):

```powershell
cd backend
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn pulse.api:app --reload
```

Open API docs at http://localhost:8000/docs. The backend accepts MP4, MOV, AVI, and MKV files up to 500 MB, with a three-minute duration limit. Data is stored locally under `backend/data` and uploads are removed after processing.

Frontend: install Node.js 20+ and pnpm, then run `cd frontend; pnpm install; pnpm dev`. It serves at http://localhost:5173. Set `VITE_API_URL` to the backend URL when it differs from localhost.

## Deploy

Deploy the **frontend** to Vercel and the **API** to a persistent Python/Docker host. The repository includes `frontend/vercel.json` and a `render.yaml`/`backend/Dockerfile` deployment path.

1. Create a backend service from this repository using `render.yaml`. Attach persistent storage at `/var/lib/pulse/data` and set `PULSE_CORS_ORIGINS` to the exact Vercel site origin (for example `https://your-project.vercel.app`). The backend stores uploaded analysis data locally, so persistent storage must be enabled.
2. In Vercel, import the same repository and set **Root Directory** to `frontend`. Vercel detects Vite; the included config builds with `pnpm build` into `dist`.
3. Add `VITE_API_URL` in Vercel's project environment variables, set to the backend origin, then redeploy.

This split is deliberate: Vercel serves the Vite UI, while the current analysis API expects local file storage and background work. Vercel's Python Functions are in beta and impose request body, duration, and bundle limits that make the current large-video upload workflow unsuitable without architectural changes. See [Vercel Python runtime](https://vercel.com/docs/functions/runtimes/python) and [Function limits](https://vercel.com/docs/functions/limitations).

## Processing methods

The sampler uses MediaPipe Face Mesh landmark regions for forehead, cheeks, and nose. Average RGB is measured per sampled frame. POS uses temporal normalization and projections in the plane orthogonal to the skin-tone direction, combined by standard deviation and overlap-add. CHROM uses normalized chrominance projections with standard-deviation weighting. Signals are band-pass filtered with a third-order Butterworth filter (0.7–4 Hz), then measured with Welch PSD. Missing measurements remain missing in stored RGB data; gap interpolation is limited to transform/filter computation and coverage affects quality.

The best ROI is selected from the measured quality score only when it exceeds the threshold. Low signal quality means insufficient physiological signal, not manipulation. rPPG is sensitive to motion, lighting, compression, skin exposure, and camera characteristics.

## API

- `POST /api/analyze/video` multipart `video` upload; returns an analysis ID immediately.
- `GET /api/analyze/{id}/status`, `/results`, `/signals`, `/report?format=json|pdf`
- `GET /api/history`, `GET /api/health`
- `POST /api/roi/compare`, `GET /api/classifier/status`
- `POST /api/classifier/train` currently returns 501 until labelled dataset ingestion and validation are implemented.

## Classifier and datasets

Classifier training is intentionally unavailable in this release. A research-grade implementation requires labelled videos, subject identifiers for leakage-resistant splits, extraction of per-video features, model calibration, and held-out evaluation. No Celeb-DF, FaceForensics++, or DFDC data is downloaded automatically. Do not treat signal disruption or poor quality as evidence of manipulation.

## Security and limitations

This is a local research prototype. Configure CORS through `PULSE_CORS_ORIGINS`, upload limit through `PULSE_MAX_UPLOAD_MB`, and storage through `PULSE_DATA_DIR`. Files are locally stored, not publicly served. Multi-face footage currently selects the largest face deterministically. The report exports measured summary JSON/PDF; model evaluation, subject selection, richer charting, and publication-grade report plots remain future work. Face detection requires the optional MediaPipe runtime to load successfully.
