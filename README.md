# DeepFake Video Forensics

A research/demo web application that screens an uploaded video using face detection and a binary Xception classifier. It reports the model's aggregate prediction and score; it cannot establish authenticity and may produce false positives or false negatives.

## Features

- Flask JSON API with health and video analysis endpoints.
- RetinaFace face detection on every 10th decoded frame, with 256 × 256 face crops.
- Existing Keras Xception checkpoint support and video-level mean-score aggregation.
- Responsive vanilla HTML/CSS/JavaScript interface with drag/drop, browser preview, light/dark themes, and accessible status messages.
- Configurable model path, upload limit, duration limit, frame interval, and CORS origin.
- Temporary uploads use unique names and are removed after inference.

## Screenshots

No screenshots are committed yet. Add reviewed screenshots under `docs/screenshots/` and link them here when available.

## Architecture

```mermaid
flowchart LR
  Browser[Vanilla JS frontend] -->|multipart video| API[Flask API]
  API --> Temp[Temporary upload]
  Temp --> CV[OpenCV frame sampling]
  CV --> Face[RetinaFace face detection]
  Face --> Prep[RGB crop, 256x256, /255]
  Prep --> Model[Keras Xception checkpoint]
  Model --> Agg[Mean sigmoid score]
  Agg --> JSON[Prediction and metadata JSON]
  JSON --> Browser
```

## Repository structure

```text
.
├── backend/
│   ├── app.py              # Flask routes and request handling
│   ├── config.py           # Environment-backed runtime settings
│   └── inference.py        # Model lifecycle and video inference
├── frontend/
│   ├── index.html
│   ├── styles.css
│   ├── app.js
│   └── config.js            # Optional API origin override
├── app.py                  # Compatibility launcher: python app.py
├── preprocess.py           # Dataset face extraction
├── split.py                # Dataset quality filtering and splits
├── train.py                # Xception training
├── evaluate.py             # Test-set evaluation
├── main.py                 # Preprocessing progress monitor
├── requirements.txt
└── README.md
```

Datasets, checkpoints, uploads, environments, and generated evaluation output are local artifacts and are not stored in this repository. Training scripts retain their root paths and layout.

## Verified technology and model

- Python backend: Flask, OpenCV, Pillow, NumPy, TensorFlow/Keras, RetinaFace.
- Frontend: HTML5, CSS3, and browser JavaScript; no frontend build step.
- Model in `train.py`: ImageNet-initialized Xception base, global average pooling, dropout 0.5, one sigmoid output, input size 256 × 256. The final 20 base-model layers are trainable in the training script.
- The training data generator maps class folders `real` to 0 and `fake` to 1. Inference follows the same mapping: sigmoid score >= 0.5 is `Fake`.
- Each detected face is independently scored; the video score is the arithmetic mean of the detected-face scores. The displayed confidence is `max(mean, 1 - mean)` as a percentage. This is a model score, not a calibrated probability.
- The API samples every 10th decoded frame (configurable with `FRAME_INTERVAL`). This is the behavior of the former web inference route. The offline dataset preprocessing script separately samples every 15th frame.

## Requirements and installation

Use Python 3.8–3.10 for the pinned TensorFlow 2.10 stack. A compatible TensorFlow build and native dependencies for RetinaFace/OpenCV are required. GPU support is optional; CPU inference may be slow. The current environment's project virtualenv is Python 3.10.10.

### Windows PowerShell

```powershell
git clone https://github.com/Samarth201104/DeepFake_Video.git
Set-Location DeepFake_Video
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS/Linux

```bash
git clone https://github.com/Samarth201104/DeepFake_Video.git
cd DeepFake_Video
python3.10 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The repository's pinned requirements include training and evaluation packages in addition to the web runtime. TensorFlow 2.10 and `tensorflow-gpu` are pinned for the original training environment and may require platform-specific installation adjustments, especially on newer Python versions and macOS. Do not install a GPU build unless the matching CUDA/cuDNN runtime is available.

## Model weights

The checkpoint `xception_best.h5` is ignored by Git because it is a large generated/model artifact. Obtain the authorized checkpoint separately and put it at the repository root, or point `DEEPFAKE_MODEL_PATH` to its location. The server starts without weights so the UI and health endpoint remain available; `/api/detect` returns HTTP 503 with a safe error until weights are configured. Never commit private or unlicensed weights.

## Run locally

From the repository root, activate the environment and run:

```powershell
python app.py
```

Or run `python -m backend.app`. Open <http://127.0.0.1:5000>. The frontend is served by the same Flask process, so the default API base URL is same-origin. To host the frontend separately, set `window.DEEPFAKE_API_BASE` in `frontend/config.js` and set `FRONTEND_ORIGIN` on the backend to that exact origin.

### Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `DEEPFAKE_MODEL_PATH` | `<repo>/xception_best.h5` | Checkpoint location |
| `MAX_UPLOAD_MB` | `500` | Maximum multipart request size |
| `MAX_VIDEO_SECONDS` | `600` | Maximum decoded-video duration |
| `FRAME_INTERVAL` | `10` | Analyze one frame per interval |
| `FRONTEND_ORIGIN` | `http://127.0.0.1:5000` | Allowed separate frontend origin |
| `HOST` | `127.0.0.1` | Bind address |
| `PORT` | `5000` | HTTP port |

For Windows PowerShell, set a variable for the current session with `$env:MAX_UPLOAD_MB = "250"`. For macOS/Linux use `export MAX_UPLOAD_MB=250`. The duration cap is checked when OpenCV can read the video's duration metadata.

## API

### `GET /api/health`

Reports server status, whether configured model weights exist, limits, and accepted filename extensions.

```json
{
  "status": "ok",
  "model_configured": true,
  "max_upload_mb": 500,
  "max_duration_seconds": 600,
  "supported_formats": [".avi", ".mkv", ".mov", ".mp4", ".webm"]
}
```

### `POST /api/detect`

Send multipart form data with a `file` field. The accepted extensions are MP4, MOV, AVI, MKV, and WebM. Upload size defaults to 500 MB; duration defaults to 10 minutes.

```bash
curl -F "file=@sample.mp4" http://127.0.0.1:5000/api/detect
```

Successful response (`200`):

```json
{
  "prediction": "Fake",
  "confidence": 87.45,
  "fake_score": 87.45,
  "confidence_type": "model score; not a calibrated probability",
  "filename": "sample.mp4",
  "duration_seconds": 12.4,
  "processed_frames": 8,
  "faces_analyzed": 7,
  "processing_seconds": 9.32
}
```

Errors return JSON with an `error` string. Common status codes are `400` for missing/empty file, `413` for oversized upload, `415` for an unsupported extension, `422` for an unreadable video or no detected faces, and `503` when model weights are unavailable. Inference progress is a single request/response; the frontend shows indeterminate activity rather than invented percentage progress.

## Dataset, training, and evaluation

These workflows are retained and require local datasets that are not included:

1. Place source videos in the folders expected by `preprocess.py` (`dataset/fake`; adapt the script for real videos as needed).
2. Run `python preprocess.py` to extract face crops. The current script writes `preprocessed/fake` and uses RetinaFace.
3. `split.py` expects `preprocessed/real` and `preprocessed/fake`, then writes `preprocessed_split/<label>/<train|val|test>`.
4. `train.py` expects `preprocessed_split/train`, `val`, and `test`, each containing `real/` and `fake/`, and writes `xception_best.h5`.
5. `evaluate.py` expects `preprocessed_split/test/{real,fake}` and writes plots under `evaluation_results/`.

The source scripts have legacy assumptions (including fake-only preprocessing and a monitor path that expects `dataset/fake`). Also, `split.py` writes `preprocessed_split/<label>/<train|val|test>`, while `train.py` and `evaluate.py` read `<split>/<label>`; reorganize the generated folders or adjust those scripts before running training/evaluation.

## Validation

Run syntax checks with `python -m compileall backend app.py`. With dependencies installed, start the app and request `GET /api/health`. `POST /api/detect` requires both valid model weights and a suitable video for end-to-end inference. The repository does not include test videos or weights in Git, so model prediction quality cannot be verified from a fresh clone without these assets.

## Troubleshooting

- **Model weights unavailable:** place the checkpoint at `xception_best.h5` or set `DEEPFAKE_MODEL_PATH`; check health JSON.
- **TensorFlow installation/import errors:** use Python 3.8–3.10 with a TensorFlow 2.10 compatible environment and adjust GPU/CUDA packages for the host OS.
- **No faces found:** try a clear video with visible, sufficiently large faces; only sampled frames are checked.
- **Video cannot be decoded:** install codecs supported by the local OpenCV build and re-encode the video to MP4/H.264.
- **CORS error:** same-origin hosting is simplest. For separate hosting, configure `FRONTEND_ORIGIN` to exactly match the browser origin.
- **Memory or long processing time:** lower `MAX_UPLOAD_MB`, `MAX_VIDEO_SECONDS`, or increase `FRAME_INTERVAL`; RetinaFace and the classifier run on CPU unless configured for a compatible GPU.

## Privacy and deployment

Uploads are written to a unique system temporary file and removed when the request ends. The service does not persist upload history. Use HTTPS and a trusted reverse proxy in deployment, keep the service bound to a private interface where appropriate, set a restrictive `FRONTEND_ORIGIN`, and configure limits for available memory and request timeouts. The built-in Flask development server is for local development, not production. Consider rate limiting and operational monitoring before exposing an instance publicly.

## Limitations and future work

The detector analyzes sampled face crops and does not provide temporal localization, per-frame results, explainability maps, or a calibrated authenticity probability. Videos without detectable faces cannot be classified. Useful future work includes automated API tests with fixture videos, a maintained model artifact distribution process, job-based processing for long videos, and broader validation across manipulation methods.

## Attribution

Repository: [Samarth201104/DeepFake_Video](https://github.com/Samarth201104/DeepFake_Video). The implementation builds on Xception and RetinaFace; consult their respective publications and licenses when redistributing or deploying model components.
