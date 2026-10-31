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

git ls-remote --exit-code origin refs/heads/main >nul 2>nul
if not errorlevel 1 (
  echo.
  echo The remote main branch already contains the uploaded project.
  echo This local checkout has a separate history because the first upload
  echo was completed through the GitHub web interface.
  echo Do not force-push this checkout. Reconcile the histories first, then
  echo run git push manually after reviewing the result.
  pause
  exit /b 2
)

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
