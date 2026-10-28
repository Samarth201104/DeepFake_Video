# DeepFake Video Forensics

A research/demo application that screens video with RetinaFace face detection and a binary Xception classifier. It provides an AI-assisted model signal; it cannot prove that a video is authentic and may produce false positives or false negatives.

## Features

- Flask JSON API with health and video analysis endpoints.
- RetinaFace face detection on every 10th decoded frame, with RGB crops resized to 256 × 256.
- The existing Keras Xception checkpoint and mean-score video aggregation.
- Responsive HTML, CSS, and vanilla JavaScript frontend with upload, preview, result display, and light/dark themes.
- Configurable checkpoint, upload size, video duration, frame interval, and CORS origin.
- Temporary upload files are deleted after each request.

## Technology and model

- Python 3.10, Flask, TensorFlow/Keras 2.10, RetinaFace, OpenCV, Pillow, and NumPy.
- HTML5, CSS3, and browser JavaScript; no frontend build system.
- `scripts/train.py` builds an ImageNet-initialized Xception base with global average pooling, 0.5 dropout, and a one-unit sigmoid output. Inputs are 256 × 256 RGB crops. The final 20 base layers are trainable.
- Class order is `real` then `fake`, mapped to 0 and 1. Inference averages detected-face sigmoid scores. An average at or above 0.5 maps to `Fake`; displayed confidence is `max(score, 1-score) * 100`. This is not a calibrated probability.
- The web pipeline samples every 10th decoded frame. Dataset preparation separately samples every 15th frame. Both use the shared RetinaFace face extraction and 256px resizing implementation; training and inference share pixel scaling to `[0,1]`.

## Repository structure

```text
.
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── api/routes.py
│   │   ├── models/model_loader.py
│   │   ├── services/detection_service.py
│   │   ├── vision/face_extractor.py
│   │   ├── vision/preprocessing.py
│   │   ├── vision/video_processor.py
│   │   ├── config.py
│   │   └── main.py
│   ├── tests/test_api.py
│   ├── tests/test_scripts.py
│   └── .env.example
├── frontend/
│   ├── index.html
│   └── assets/{css,js}/
├── scripts/
│   ├── prepare_dataset.py
│   ├── split_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   └── monitor_preprocessing.py
├── data/
│   ├── raw/fake/.gitkeep    # place source fake videos here
│   ├── processed/
│   │   ├── faces/           # preprocessed real/fake face crops
│   │   └── splits/          # train/val/test folders
│   └── uploads/
│       ├── api/             # preserved prior upload samples
│       └── static/          # preserved legacy static upload samples
├── artifacts/
│   ├── models/              # local checkpoints, ignored by Git
│   └── evaluation/          # generated plots, ignored by Git
├── .venv/                   # local Python environment, ignored by Git
├── .gitignore
├── requirements.txt
└── README.md
```

Raw videos belong in `data/raw/fake/`; extracted crops and splits go under `data/processed/`; existing uploaded videos were preserved under `data/uploads/api/` and `data/uploads/static/`; checkpoints and reports go under `artifacts/`. These local artifacts and `.venv/` are ignored by Git. The historical root `app.py`, `main.py`, and training scripts were moved to their single maintained locations; no alternate app launcher remains.

## Installation

TensorFlow 2.10 is compatible with Python 3.7–3.10. Python 3.10 is recommended. GPU support is optional and requires a compatible TensorFlow/CUDA environment.

### Windows PowerShell

```powershell
git clone https://github.com/Samarth201104/DeepFake_Video.git
Set-Location DeepFake_Video
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS/Linux

```bash
git clone https://github.com/Samarth201104/DeepFake_Video.git
cd DeepFake_Video
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The compact requirements file lists direct runtime/training dependencies rather than all transitive packages. `retina-face` may install or require additional packages for its selected environment. On platforms where TensorFlow 2.10 wheels are unavailable, use a compatible Python 3.10 environment; do not blindly upgrade TensorFlow because the checkpoint and model pipeline were built against the existing stack.

## Weights and configuration

The trained `xception_best.h5` checkpoint is not tracked because it is a large local artifact. Place the authorized checkpoint at `artifacts/models/xception_best.h5` or set `DEEPFAKE_MODEL_PATH` to its path. The server starts without it, health reports whether it exists, and detection returns HTTP 503 until it is configured.

`backend/.env.example` documents supported environment variables. The app reads process environment variables directly; it does not load `.env` files. Copy values into your shell or a deployment environment:

| Variable | Default | Purpose |
| --- | --- | --- |
| `DEEPFAKE_MODEL_PATH` | `<repo>/artifacts/models/xception_best.h5` | Model checkpoint |
| `MAX_UPLOAD_MB` | `500` | Maximum multipart request size |
| `MAX_VIDEO_SECONDS` | `600` | Maximum decoded-video duration |
| `FRAME_INTERVAL` | `10` | Analyze one frame per interval |
| `FRONTEND_ORIGIN` | `http://127.0.0.1:5000` | Allowed separate frontend origin |
| `HOST` | `127.0.0.1` | Bind address |
| `PORT` | `5000` | Server port |

For PowerShell, for example: `$env:MAX_UPLOAD_MB = "250"`. For macOS/Linux: `export MAX_UPLOAD_MB=250`.

## Run the application

From the repository root with the environment activated:

```bash
python -m backend.app.main
```

Open <http://127.0.0.1:5000>. Flask serves `frontend/index.html` and `/assets/...` from the same origin. If serving the frontend separately, set `window.DEEPFAKE_API_BASE` in `frontend/assets/js/config.js` and set the backend `FRONTEND_ORIGIN` to the exact frontend origin.

## API

### `GET /api/health`

Returns service status, model presence, configured limits, and accepted filename extensions.

### `POST /api/detect`

Send a multipart form upload in the `file` field. Supported filename extensions: MP4, MOV, AVI, MKV, and WebM. Defaults are 500 MB and 10 minutes.

```bash
curl -F "file=@sample.mp4" http://127.0.0.1:5000/api/detect
```

Successful JSON includes `prediction`, `confidence`, `fake_score`, `confidence_type`, `filename`, `duration_seconds`, `processed_frames`, `faces_analyzed`, and `processing_seconds`. Errors return JSON with an `error` field: 400 for a missing/empty upload, 413 for size limit, 415 for an unsupported extension, 422 for unreadable/no-face videos, and 503 for unavailable model weights.

## Dataset preparation, training, and evaluation

Place source fake videos under `data/raw/fake/`; the current preparation script extracts fake faces only. Create `data/processed/faces/real/` face crops separately before splitting. Commands run from the repository root:

```bash
python -m scripts.prepare_dataset
python -m scripts.split_dataset
python -m scripts.train
python -m scripts.evaluate
```

`prepare_dataset` samples every 15th frame and saves RetinaFace crops to `data/processed/faces/fake/`. It supports `--input-dir`, `--output-dir`, `--frame-interval`, `--target-faces`, and an optional `--seed` for video order. `split_dataset` supports `--input-dir`, `--output-dir`, and `--seed` (default 42), quality-filters images (minimum 30px and Laplacian variance >=20), partitions them 70/15/15, and writes the folder structure expected by the training and evaluation scripts:

```text
data/processed/splits/{train,val,test}/{real,fake}/
```

Training parameters and augmentation remain the existing values: 256px input, batch 16, 50 maximum epochs, Adam at 1e-4, binary cross entropy, mixed precision, early stopping, and learning-rate reduction. Training saves `artifacts/models/xception_best.h5`. Evaluation reads that checkpoint and `data/processed/splits/test/`, then saves plots in `artifacts/evaluation/`.

Monitor extraction separately with `python -m scripts.monitor_preprocessing` (Ctrl+C to stop). The scripts use paths relative to the repository, not the current shell directory.

## Tests and validation

Run the API and dataset split checks with:

```bash
python -m unittest discover -s backend/tests -v
```

Check syntax with `python -m compileall backend scripts`. Full inference requires the checkpoint and a suitable face-containing test video. Training/evaluation require the local prepared dataset and are not run as part of application startup.

## Troubleshooting

- **Weights missing/unreadable:** place the checkpoint at the configured path and verify `/api/health`.
- **TensorFlow import/install errors:** use Python 3.10 and a compatible TensorFlow 2.10 platform build.
- **Video decode failures:** use a codec supported by the local OpenCV build, such as H.264 MP4.
- **No detected faces:** use a clear clip with visible faces; only sampled frames are inspected.
- **CORS failure:** same-origin hosting is simplest. For separate hosting, match `FRONTEND_ORIGIN` exactly to the browser origin.
- **Memory/time pressure:** lower upload/duration limits or increase `FRAME_INTERVAL`; RetinaFace and model inference may be CPU-bound.

## Limitations and future improvements

The application produces a video-level model signal from sampled face crops. It does not localize manipulated frames, expose explainability maps, or return calibrated probabilities. It cannot classify videos without detectable faces. Future work could add maintained model artifact distribution, dataset-aware integration tests, and asynchronous processing for long videos.

## Privacy and deployment

The backend writes uploads to unique temporary files and deletes them after inference. Use HTTPS, a production WSGI server, request timeouts, suitable resource limits, and a restrictive CORS origin before public deployment. The built-in Flask server is for local development.

Project: [Samarth201104/DeepFake_Video](https://github.com/Samarth201104/DeepFake_Video).
