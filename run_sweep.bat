@echo off
rem Offline synthetic-data parameter sweep. A full pass is only an internal
rem screening result; it is not an RF, covertness, or OTA validation.
cd /d "%~dp0"
set PY=python
set CKPT=lpi_checkpoint_v3_610.pt

"%PY%" sweep.py --ckpt "%CKPT%" --chunk 2 --max-epoch 760 --n-samples 50000
if errorlevel 1 (
    echo Sweep command failed. Check the Python environment and logs.
    exit /b 1
)

if exist "%CKPT%.best.pt" (
    echo Offline screening pass found; exporting the checkpoint.
    copy /y "%CKPT%.best.pt" lpi_checkpoint.pt
    "%PY%" test_metrics.py --ckpt lpi_checkpoint.pt --json-out lpi_metrics.json
    if errorlevel 1 exit /b 1
    "%PY%" export_for_grc.py --checkpoint lpi_checkpoint.pt
) else (
    echo No offline checkpoint passed all configured screening checks up to max epoch.
    echo Review the sweep CSV; no checkpoint was selected or exported.
)
