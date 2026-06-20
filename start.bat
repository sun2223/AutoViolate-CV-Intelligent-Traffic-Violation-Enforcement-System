@echo off
echo Starting AutoViolate-CV Backend...
start cmd /k "python -m uvicorn app:app --host 0.0.0.0 --port 8000"

echo Starting AutoViolate-CV Frontend...
cd frontend
start cmd /k "npm run dev"

echo Both services are starting!
echo Backend API will be at: http://localhost:8000
echo Frontend UI will be at: http://localhost:3000
