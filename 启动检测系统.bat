@echo off
setlocal
cd /d "%~dp0"

echo ================================================
echo   Forest smoke and flame detection system
echo ================================================
echo.
echo Starting the local web app. Model loading may take a few seconds.
echo Keep the server window open while using the system.
echo.

start "Forest Smoke Server" /min "%~dp0run_streamlit_public.bat"

set "PORT=8501"
for /l %%N in (1,1,30) do (
  if exist "%~dp0.forest_smoke_public_port" set /p PORT=<"%~dp0.forest_smoke_public_port"
  netstat -ano | findstr /R /C:":%PORT% .*LISTENING" >nul
  if not errorlevel 1 goto OPEN_BROWSER
  timeout /t 1 /nobreak >nul
)

echo.
echo [START FAILED] The local service did not start in time.
echo Check the minimized server window for the detailed error.
pause
exit /b 1

:OPEN_BROWSER
echo Local system is ready: http://127.0.0.1:%PORT%/
start "" "http://127.0.0.1:%PORT%/"
exit /b 0
