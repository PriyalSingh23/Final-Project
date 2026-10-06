@echo off
rem LPI-CGAN overnight sweep: trains + tests in a loop until a snapshot
rem passes ALL metrics, then exports it for GNU Radio.
rem Safe to double-click; safe to close mid-run (progress is saved every epoch).
cd /d "%~dp0"
set PY=C:\Users\yasht\radioconda\python.exe
set CKPT=lpi_checkpoint_v3_610.pt
set GOLD=GOLD_XYZ_OSC.0001_1024.hdf5\GOLD_XYZ_OSC.0001_1024.hdf5

"%PY%" sweep.py --ckpt %CKPT% --chunk 2 --max-epoch 760 --n-samples 50000 --data "%GOLD%"
if exist "%CKPT%.best.pt" (
    echo *** Full-PASS snapshot found -- exporting for GNU Radio ***
    copy /y "%CKPT%.best.pt" lpi_checkpoint.pt
    "%PY%" export_for_grc.py
    "%PY%" test_metrics.py --ckpt lpi_checkpoint.pt
) else (
    echo No full-PASS snapshot up to max epoch. Resume later with:
    echo   %PY% sweep.py --ckpt %CKPT% --chunk 2 --max-epoch 860 --n-samples 50000 --data "%GOLD%"
)
pause
