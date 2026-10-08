@echo off
setlocal
echo =========================================================================
echo PROJECT ANSHUMAN -- LAUNCH GNU RADIO COMPANION WITH LPI-CGAN FLOWGRAPHS
echo =========================================================================
echo.

set PATH=C:\Users\gspra\radioconda\Library\bin;C:\Users\gspra\radioconda\Scripts;C:\Users\gspra\radioconda;%PATH%
set PYTHONPATH=C:\Users\gspra\radioconda\Lib\site-packages;c:\Users\gspra\OneDrive\Desktop\LPI_CGAN;c:\Users\gspra\OneDrive\Desktop\LPI_CGAN\grc;%PYTHONPATH%

echo Opening Transmitter and Receiver flowgraphs in GNU Radio Companion...
start "" "C:\Users\gspra\radioconda\Scripts\gnuradio-companion.exe" --qt "c:\Users\gspra\OneDrive\Desktop\LPI_CGAN\grc\tx_lpi_cgan.grc" "c:\Users\gspra\OneDrive\Desktop\LPI_CGAN\grc\rx_lpi_cgan.grc"

echo Flowgraphs opened in GNU Radio Companion.
