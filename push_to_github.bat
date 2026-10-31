@echo off
setlocal
cd /d "%~dp0"

echo ================================================
echo  Push forest fire detection project to GitHub
echo ================================================
echo Repository: https://github.com/amirex-creator/forest-fire-yolo-detection
echo.
echo A GitHub sign-in window or credential prompt may appear.
echo Do not paste a password into the repository files.
echo.

set "GIT_TERMINAL_PROMPT=1"
set "GCM_INTERACTIVE=always"
git config http.proxy http://127.0.0.1:7897
git config https.proxy http://127.0.0.1:7897
git config http.sslBackend openssl

git push -u origin main
if errorlevel 1 (
  echo.
  echo Push failed. Check the GitHub sign-in message above.
  echo The local commit is still available and can be pushed later.
) else (
  echo.
  echo Push completed. Open the repository URL above to verify the files.
)
pause
