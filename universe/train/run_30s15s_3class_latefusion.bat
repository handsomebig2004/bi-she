@echo off
setlocal EnableExtensions

cd /d "%~dp0\..\.."

set "PYTHON=D:\anaconda3\envs\proj\python.exe"
set "DATA_ROOT=data\UNIVERSE"
set "OUT_ROOT=results\cnn_window_ablation"
set "LOG_DIR=%OUT_ROOT%\logs"

set "WIN_S=30"
set "HOP_S=15"
set "TAG=30s15s"
set "EDA_DIR=%DATA_ROOT%\windows_eda_%TAG%"
set "BVP_DIR=%DATA_ROOT%\windows_bvp_%TAG%"

set "EPOCHS=30"
set "LR=0.001"
set "BATCH_SIZE=64"
set "DROPOUT=0.25"

if not exist "%PYTHON%" (
    echo Python executable not found: %PYTHON%
    exit /b 1
)

if not exist "%OUT_ROOT%" mkdir "%OUT_ROOT%"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo ============================================================
echo UNIVERSE 30s/15s 3-class late-fusion experiments
echo EDA windows: %EDA_DIR%
echo BVP windows: %BVP_DIR%
echo Results:     %OUT_ROOT%\%TAG%
echo Logs:        %LOG_DIR%
echo ============================================================

if not exist "%EDA_DIR%\X.npy" (
    echo Building EDA windows: %EDA_DIR%
    "%PYTHON%" universe\preprocess\eda_build_windows.py ^
        --win-s %WIN_S% ^
        --hop-s %HOP_S% ^
        --out-dir "%EDA_DIR%" ^
        > "%LOG_DIR%\build_eda_%TAG%.log" 2>&1
    if errorlevel 1 goto failed
)

if not exist "%BVP_DIR%\X.npy" (
    echo Building BVP windows: %BVP_DIR%
    "%PYTHON%" universe\preprocess\bvp_build_windows.py ^
        --win-s %WIN_S% ^
        --hop-s %HOP_S% ^
        --out-dir "%BVP_DIR%" ^
        > "%LOG_DIR%\build_bvp_%TAG%.log" 2>&1
    if errorlevel 1 goto failed
)

echo.
echo [1/2] LateFusion CNN 3-class, %TAG%
"%PYTHON%" universe\train\cnn_eda_bvp_latefusion_loso.py ^
    --eda-dir "%EDA_DIR%" ^
    --bvp-dir "%BVP_DIR%" ^
    --epochs %EPOCHS% ^
    --lr %LR% ^
    --batch-size %BATCH_SIZE% ^
    --dropout %DROPOUT% ^
    --out-dir "%OUT_ROOT%\%TAG%\latefusion_cnn" ^
    > "%LOG_DIR%\%TAG%_latefusion_cnn_3class.log" 2>&1
if errorlevel 1 goto failed

echo.
echo [2/2] LateFusion ResNet 3-class, %TAG%
"%PYTHON%" universe\train\cnn_eda_bvp_latefusion_resnet_loso.py ^
    --eda-dir "%EDA_DIR%" ^
    --bvp-dir "%BVP_DIR%" ^
    --epochs %EPOCHS% ^
    --lr %LR% ^
    --batch-size %BATCH_SIZE% ^
    --dropout %DROPOUT% ^
    --out-dir "%OUT_ROOT%\%TAG%\latefusion_resnet" ^
    > "%LOG_DIR%\%TAG%_latefusion_resnet_3class.log" 2>&1
if errorlevel 1 goto failed

echo.
echo Updating collected summaries
"%PYTHON%" universe\visual\collect_window_ablation_results.py ^
    --root "%OUT_ROOT%" ^
    --out "%OUT_ROOT%\summary_all.csv" ^
    > "%LOG_DIR%\collect_summary.log" 2>&1
if errorlevel 1 goto failed

"%PYTHON%" universe\visual\collect_all_cnn_results.py ^
    > "%LOG_DIR%\collect_all_cnn_results.log" 2>&1
if errorlevel 1 goto failed

echo.
echo ============================================================
echo 30s/15s 3-class experiments finished.
echo Result files:
echo   %OUT_ROOT%\%TAG%\latefusion_cnn\loso_results_kept.csv
echo   %OUT_ROOT%\%TAG%\latefusion_cnn\loso_results_skipped.csv
echo   %OUT_ROOT%\%TAG%\latefusion_resnet\loso_results_kept.csv
echo   %OUT_ROOT%\%TAG%\latefusion_resnet\loso_results_skipped.csv
echo Updated summaries:
echo   %OUT_ROOT%\summary_all.csv
echo   results\cnn_collected\cnn_summary_all.csv
echo ============================================================
exit /b 0

:failed
echo.
echo 30s/15s 3-class experiment failed. Check logs in %LOG_DIR%.
exit /b 1
