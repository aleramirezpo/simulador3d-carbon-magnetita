@echo off
setlocal
cd /d "%~dp0.."
python -m interfaz.app
if errorlevel 1 pause

