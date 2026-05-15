@echo off
setlocal EnableExtensions

cd /d "%~dp0\.."

set "PYTHON=D:\anaconda3\envs\proj\python.exe"
set "DATA_DIR=data\MAUS\windows_gsr_ppg_30s15s_kmeans2"
set "OUT_DIR=results\maus_latefusion_resnet_binary_loso"
set "LOG_DIR=%OUT_DIR%\logs"

if not exist "%PYTHON%" (
    echo Python executable not found: %PYTHON%
    exit /b 1
)

if not exist "%DATA_DIR%\X_gsr.npy" (
    echo Missing MAUS windows. Building them first...
    "%PYTHON%" maus\build_windows_kmeans_binary.py ^
        --out-dir "%DATA_DIR%" ^
        > "%OUT_DIR%_build_windows.log" 2>&1
    if errorlevel 1 goto failed
)

if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo ============================================================
echo MAUS LateFusion ResNet binary LOSO
echo Data:    %DATA_DIR%
echo Results: %OUT_DIR%
echo ============================================================

"%PYTHON%" maus\train_latefusion_resnet_binary_loso.py ^
    --data-dir "%DATA_DIR%" ^
    --out-dir "%OUT_DIR%" ^
    --config-name "maus_loso_30s15s" ^
    --run-seed 0 ^
    --epochs 40 ^
    --lr 0.0003 ^
    --weight-decay 0.0001 ^
    --dropout 0.25 ^
    --batch-size 64 ^
    --scheduler ^
    --early-stop-patience 8 ^
    > "%LOG_DIR%\maus_loso_30s15s.log" 2>&1
if errorlevel 1 goto failed

echo.
echo Finished.
echo Result files:
echo   %OUT_DIR%\loso_results_kept.csv
echo   %OUT_DIR%\loso_results_skipped.csv
echo Log:
echo   %LOG_DIR%\maus_loso_30s15s.log
exit /b 0

:failed
echo.
echo MAUS LOSO failed. Check logs.
exit /b 1
