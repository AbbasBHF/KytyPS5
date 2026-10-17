@echo off
cd /d "%~dp0"
cmake --install _Build/windows --prefix _Build/windows/install
exit /b %ERRORLEVEL%
