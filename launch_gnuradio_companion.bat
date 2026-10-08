@echo off
setlocal
echo =========================================================================
echo PROJECT ANSHUMAN -- LAUNCH GNU RADIO COMPANION WITH LPI-CGAN FLOWGRAPHS
echo =========================================================================
echo.

set GRC=C:\Users\gspra\radioconda\Scripts\gnuradio-companion.exe

echo Opening Transmitter and Receiver flowgraphs in GNU Radio Companion...
start "" "%GRC%" "c:\Users\gspra\OneDrive\Desktop\LPI_CGAN\grc\tx_lpi_cgan.grc" "c:\Users\gspra\OneDrive\Desktop\LPI_CGAN\grc\rx_lpi_cgan.grc"

echo Flowgraphs opened in GNU Radio Companion.
