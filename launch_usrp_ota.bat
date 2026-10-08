@echo off
setlocal
echo =========================================================================
echo PROJECT ANSHUMAN -- USRP REAL-TIME OVER-THE-AIR LPI-CGAN TRANSCEIVER
echo =========================================================================
echo.

set PYTHON=C:\Users\gspra\radioconda\python.exe
set UHD_FIND=C:\Users\gspra\radioconda\Library\bin\uhd_find_devices.exe

echo [1/3] Checking UHD Driver and Connected USRP Hardware...
%UHD_FIND%
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [NOTICE] USRP driver not yet bound to WinUSB.
    echo Please run Zadig (already open in Downloads) and install WinUSB for 'WestBridge'.
    echo.
    pause
    exit /b 1
)

echo.
echo [2/3] Executing Physical Real-Time OTA Transceiver Check...
%PYTHON% lpi_research\sdr\usrp_transceiver.py --msg "USRP_LPI_SECURE_01" --freq 750e6 --rate 1e6 --tx-gain 25 --rx-gain 35

echo.
echo [3/3] OTA Transceiver Check Finished.
pause
