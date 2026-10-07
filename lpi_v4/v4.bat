@echo off
REM ===========================================================================
REM  ShadowComm LPI v4 - one entry point for everything on Windows.
REM  Usage:  lpi_v4\v4.bat <command> [extra args...]
REM
REM    test   pytest loopback + framing tests (no GPU, no radio)
REM    check  check_docs.py          : do the documented commands still parse?
REM    loop   rx_usrp.py --selftest  : TX -> channel -> RX -> AES/RS -> text
REM    grc    lpi_grc.py             : exercises the blocks the .grc files embed
REM    train  lpi_train.py           : add --epochs 200 etc. to override
REM    eval   lpi_eval.py            : the field-gate report (exit 0 == all pass)
REM    tx     tx_usrp.py             : real-time transmit (--addr 192.168.10.2)
REM    rx     rx_usrp.py             : real-time receive  (--watch --gain 30)
REM    info   uhd_find_devices + device list
REM    py     drop into the same python with PYTHONPATH set
REM
REM  PYTHON: set LPI_PYTHON to the interpreter that has torch installed, e.g.
REM    set LPI_PYTHON=C:\Users\yasht\radioconda\python.exe
REM ===========================================================================
setlocal
set HERE=%~dp0
set LPI_PYTHON=%LPI_PYTHON%
if "%LPI_PYTHON%"=="" set LPI_PYTHON=python
set PYTHONPATH=%HERE%;%PYTHONPATH%
set OMP_NUM_THREADS=4

if /I "%1"=="test"  goto test
if /I "%1"=="check" goto check
if /I "%1"=="loop"  goto loop
if /I "%1"=="grc"   goto grc
if /I "%1"=="train" goto train
if /I "%1"=="eval"  goto eval
if /I "%1"=="tx"    goto tx
if /I "%1"=="rx"    goto rx
if /I "%1"=="info"  goto info
if /I "%1"=="py"    goto py
goto help

:test
%LPI_PYTHON% -m pytest "%HERE%..\tests" -q -m "not slow" %2 %3 %4
goto end

:check
%LPI_PYTHON% "%HERE%check_docs.py" %2 %3 %4
goto end

:loop
%LPI_PYTHON% "%HERE%rx_usrp.py" --selftest %2 %3 %4
goto end

:grc
%LPI_PYTHON% "%HERE%lpi_grc.py" %2 %3 %4
goto end

:train
%LPI_PYTHON% "%HERE%lpi_train.py" --out "%HERE%run\lpi_v4.pt" --log "%HERE%run\lpi_v4.csv" %2 %3 %4
goto end

:eval
%LPI_PYTHON% "%HERE%lpi_eval.py" --ckpt "%HERE%run\lpi_v4.best.pt" --fit-warden 800 --md "%HERE%run\eval.md" --out-json "%HERE%run\eval.json" %2 %3 %4
goto end

:tx
%LPI_PYTHON% "%HERE%tx_usrp.py" %2 %3 %4 %5
goto end

:rx
%LPI_PYTHON% "%HERE%rx_usrp.py" %2 %3 %4 %5
goto end

:info
where uhd_find_devices >nul 2>nul && (uhd_find_devices) || (echo uhd_find_devices not on PATH - install UHD / radioconda & %LPI_PYTHON% -c "import importlib.util as u; print('python uhd bindings:', u.find_spec('uhd') is not None)")
goto end

:py
cd /d "%HERE%"
%LPI_PYTHON%
goto end

:help
echo commands: test check loop grc train eval tx rx info py
echo (see the comments at the top of this file for what each one runs)
exit /b 2

:end
endlocal
