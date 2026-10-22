# DetectAI: Run Guide

This project has a FastAPI backend and a Vite/React frontend. The frontend proxies
API requests to the backend, so use the frontend link for normal application use.

## Application links

- **Frontend:** http://127.0.0.1:5173/
- **Backend health:** http://127.0.0.1:8000/health
- **Frontend-to-backend health check:** http://127.0.0.1:5173/health
- **Backend API docs:** http://127.0.0.1:8000/docs

## Requirements

- Windows PowerShell
- Python 3.11 or newer
- Node.js and npm

The repository already includes a Python virtual environment in `.venv` and
installed frontend dependencies in `frontend/node_modules`.

## Start the backend

Open **PowerShell 1** and run:

```powershell
Set-Location "C:\Users\nanda\OneDrive\Documents\NLP Project\ai-text-detector"
& ".\.venv\Scripts\python.exe" -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Leave this terminal running. Confirm the backend is ready:

```powershell
Invoke-WebRequest "http://127.0.0.1:8000/health" -UseBasicParsing
```

The response should contain:

```json
{"ok":true}
```

## Start the frontend

Open **PowerShell 2** and run:

```powershell
Set-Location "C:\Users\nanda\OneDrive\Documents\NLP Project\ai-text-detector\frontend"
npm.cmd run dev -- --host 127.0.0.1
```

Leave this terminal running. Then open:

**http://127.0.0.1:5173/**

The Vite proxy forwards `/predict`, `/metrics`, `/humanize`, and `/health` to
the backend at `http://127.0.0.1:8000`.

## First-time setup

If dependencies are missing, run these commands once:

```powershell
Set-Location "C:\Users\nanda\OneDrive\Documents\NLP Project\ai-text-detector"
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt

Set-Location ".\frontend"
npm.cmd install
```

## Production frontend build

To create a production frontend bundle:

```powershell
Set-Location "C:\Users\nanda\OneDrive\Documents\NLP Project\ai-text-detector\frontend"
npm.cmd run build
```

The FastAPI backend serves the generated frontend from `frontend/dist` when it
exists. Start the backend from the project root as shown above and open:

**http://127.0.0.1:8000/**

## Validate the project

Run backend tests:

```powershell
Set-Location "C:\Users\nanda\OneDrive\Documents\NLP Project\ai-text-detector"
& ".\.venv\Scripts\python.exe" -m pytest -q
```

Build the frontend:

```powershell
Set-Location ".\frontend"
npm.cmd run build
```

## Stop the development servers

Focus each terminal running a server and press:

```text
Ctrl+C
```

## Prediction notes

- Enter at least 30 words for an analysis.
- Supported uploads are `.txt`, `.md`, and `.csv`.
- The result is a supervised ML estimate and should not be treated as definitive
  proof of authorship.
- The current held-out evaluation metrics are shown in the application:
  97.8% accuracy and 99.8% ROC AUC over 4,780 test samples.

For the complete evaluation values, confusion matrix, latency measurement,
dataset counts, environment details, and figure-generation notes, see
[EVALUATION_DETAILS.md](./EVALUATION_DETAILS.md).
