@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_windows.ps1"
if errorlevel 1 (
  echo.
  echo JunQi Windows client failed to start.
  pause
  exit /b 1
)
endlocal
