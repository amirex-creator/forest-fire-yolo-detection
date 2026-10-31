@echo off
setlocal
cd /d "%~dp0"

set "PORT=8501"
if exist "%~dp0.forest_smoke_public_port" set /p PORT=<"%~dp0.forest_smoke_public_port"

netstat -ano | findstr /R /C:":%PORT% .*LISTENING" >nul
if not errorlevel 1 goto SERVICE_FOUND
set "PORT=8501"
netstat -ano | findstr /R /C:":8501 .*LISTENING" >nul
if not errorlevel 1 goto SERVICE_FOUND
set "PORT=8502"
netstat -ano | findstr /R /C:":8502 .*LISTENING" >nul
if not errorlevel 1 goto SERVICE_FOUND
echo.
echo [TUNNEL FAILED] No Streamlit service is listening on port 8501 or 8502.
echo Start the local app first by double-clicking the Chinese-named launcher in this folder.
pause
exit /b 1

:SERVICE_FOUND
>"%~dp0.forest_smoke_public_port" echo %PORT%

set "CLOUDFLARED="
where cloudflared >nul 2>nul
if not errorlevel 1 set "CLOUDFLARED=cloudflared"
if not defined CLOUDFLARED if exist "%~dp0..\tools\cloudflared.exe" set "CLOUDFLARED=%~dp0..\tools\cloudflared.exe"
if not defined CLOUDFLARED (
  echo cloudflared was not found in PATH or ..\tools\cloudflared.exe.
  echo Install it from the official Cloudflare release, then run this file again.
  pause
  exit /b 1
)

echo The Streamlit service is listening on 127.0.0.1:%PORT%.
echo Press Ctrl+C to stop the public tunnel.
"%CLOUDFLARED%" tunnel --no-autoupdate --edge-ip-version 4 --protocol http2 --url http://127.0.0.1:%PORT%
if errorlevel 1 (
  echo.
  echo [TUNNEL FAILED] Cloudflare could not establish the connection.
  echo Check outbound network access, proxy settings, and the cloudflared output above.
)
pause
