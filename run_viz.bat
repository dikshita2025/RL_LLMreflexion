@echo off
REM ─────────────────────────────────────────────────────────────────────────────
REM LLM4Teach — Live training visualizer (FastAPI + WebSocket web dashboard)
REM   Single entry point for the viz/ package. Open the printed URL, press Start,
REM   pick a phase (ppo_only / planner / reflection) or run all 3, and watch the
REM   MiniGrid grid, LLM calls, and live metric charts update in real time.
REM ─────────────────────────────────────────────────────────────────────────────
setlocal
pushd "%~dp0"

REM Use the project venv if setup.bat created one; otherwise fall back to system python.
if exist "venv\Scripts\activate.bat" call "venv\Scripts\activate.bat"

echo ============================================================
echo  MiniGrid Training Visualizer
echo  Installing viz dependencies (fastapi + uvicorn)...
echo ============================================================
pip install -r "viz\requirements_viz.txt" --quiet

echo.
echo ============================================================
echo  Starting server at http://localhost:7860
echo  Open that URL in your browser, then press Start.
echo  (Online LLM is optional: needs "ollama serve" + "ollama pull qwen2.5:3b")
echo ============================================================
python "viz\run_viz.py" %*

popd
endlocal
pause
