@echo off
setlocal
cd /d "%~dp0"

set "PORT=8501"
del /q "%~dp0.forest_smoke_public_port" >nul 2>nul
netstat -ano | findstr /R /C:":8501 .*LISTENING" >nul
if not errorlevel 1 set "PORT=8502"
>"%~dp0.forest_smoke_public_port" echo %PORT%

set "FOREST_SMOKE_PUBLIC_MODE=1"
set "FOREST_SMOKE_DEVICE=cpu"
set "FOREST_SMOKE_MODEL=models\hardneg_finetune_cont30_full_20261029_034435_best.pt"
set "YOLO_CONFIG_DIR=%~dp0outputs\ultralytics"
set "MPLCONFIGDIR=%~dp0outputs\matplotlib"

rem Prefer an environment that actually contains Streamlit.
set "PYTHON="
set "PYTHON_ARGS="
call :try_python "%~dp0.venv\Scripts\python.exe"
call :try_python "%~dp0venv\Scripts\python.exe"
call :try_python "%~dp0..\.venv\Scripts\python.exe"
call :try_python "%~dp0..\venv\Scripts\python.exe"
call :try_command python
call :try_command py

if not defined PYTHON (
  echo.
  echo [START FAILED] No Python environment with Streamlit was found.
  echo Project: %~dp0
  echo.
  echo Install once from this folder:
  echo   python -m venv .venv
  echo   .venv\Scripts\python.exe -m pip install -r requirements-deploy.txt
  echo.
  pause
  exit /b 1
)

echo Starting Streamlit on http://127.0.0.1:%PORT%/
echo Python: %PYTHON% %PYTHON_ARGS%
"%PYTHON%" %PYTHON_ARGS% -m streamlit run app.py --server.address 127.0.0.1 --server.port %PORT% --server.headless true
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" echo [START FAILED] Streamlit exit code: %EXIT_CODE%
pause
exit /b %EXIT_CODE%

:try_python
if defined PYTHON exit /b 0
if not exist "%~1" exit /b 0
"%~1" -c "import streamlit" >nul 2>nul
if not errorlevel 1 set "PYTHON=%~1"
exit /b 0

:try_command
if defined PYTHON exit /b 0
where %~1 >nul 2>nul
if errorlevel 1 exit /b 0
if /i "%~1"=="py" (
  py -3 -c "import streamlit" >nul 2>nul
  if not errorlevel 1 (
    set "PYTHON=py"
    set "PYTHON_ARGS=-3"
  )
) else (
  %~1 -c "import streamlit" >nul 2>nul
  if not errorlevel 1 set "PYTHON=%~1"
)
exit /b 0
