@echo off
cd /d "%~dp0"
cmake --build _Build/windows --target launcher --parallel
exit /b %ERRORLEVEL%
