@echo off
setlocal EnableExtensions

cd /d "%~dp0\..\.."

set "PYTHON=D:\anaconda3\envs\proj\python.exe"
set "DATA_ROOT=data\UNIVERSE"
set "OUT_ROOT=results\cnn_window_ablation"
set "LOG_DIR=%OUT_ROOT%\logs"

set "EPOCHS=30"
set "LR=0.001"
set "BATCH_SIZE=64"
set "DROPOUT=0.25"

rem First-pass ablation: binary is faster and usually more informative.
rem Set RUN_3CLASS=1 if you also want the full 3-class matrix.
set "RUN_BINARY=1"
set "RUN_3CLASS=0"
set "BUILD_WINDOWS=1"

if not exist "%PYTHON%" (
    echo Python executable not found: %PYTHON%
    exit /b 1
)

if not exist "%OUT_ROOT%" mkdir "%OUT_ROOT%"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo ============================================================
echo UNIVERSE window-length ablation for late-fusion CNN/ResNet
echo BUILD_WINDOWS=%BUILD_WINDOWS% RUN_BINARY=%RUN_BINARY% RUN_3CLASS=%RUN_3CLASS%
echo Results: %OUT_ROOT%
echo Logs:    %LOG_DIR%
echo ============================================================

call :run_setting 30 15
if errorlevel 1 goto failed

call :run_setting 60 30
if errorlevel 1 goto failed

call :run_setting 90 30
if errorlevel 1 goto failed

call :run_setting 120 60
if errorlevel 1 goto failed

echo.
echo Collecting summary CSV
"%PYTHON%" universe\visual\collect_window_ablation_results.py ^
    --root "%OUT_ROOT%" ^
    --out "%OUT_ROOT%\summary_all.csv" ^
    > "%LOG_DIR%\collect_summary.log" 2>&1
if errorlevel 1 goto failed

echo.
echo ============================================================
echo Window ablation finished.
echo Summary:
echo   %OUT_ROOT%\summary_all.csv
echo Result folders:
echo   %OUT_ROOT%\30s15s
echo   %OUT_ROOT%\60s30s
echo   %OUT_ROOT%\90s30s
echo   %OUT_ROOT%\120s60s
echo ============================================================
exit /b 0

:run_setting
set "WIN_S=%~1"
set "HOP_S=%~2"
set "TAG=%WIN_S%s%HOP_S%s"
set "EDA_DIR=%DATA_ROOT%\windows_eda_%TAG%"
set "BVP_DIR=%DATA_ROOT%\windows_bvp_%TAG%"
set "TAG_OUT=%OUT_ROOT%\%TAG%"

echo.
echo ------------------------------------------------------------
echo Window setting: %TAG%  window=%WIN_S%s hop=%HOP_S%s
echo ------------------------------------------------------------

if "%BUILD_WINDOWS%"=="1" (
    echo Building EDA windows: %EDA_DIR%
    "%PYTHON%" universe\preprocess\eda_build_windows.py ^
        --win-s %WIN_S% ^
        --hop-s %HOP_S% ^
        --out-dir "%EDA_DIR%" ^
        > "%LOG_DIR%\build_eda_%TAG%.log" 2>&1
    if errorlevel 1 exit /b 1

    echo Building BVP windows: %BVP_DIR%
    "%PYTHON%" universe\preprocess\bvp_build_windows.py ^
        --win-s %WIN_S% ^
        --hop-s %HOP_S% ^
        --out-dir "%BVP_DIR%" ^
        > "%LOG_DIR%\build_bvp_%TAG%.log" 2>&1
    if errorlevel 1 exit /b 1
)

if "%RUN_BINARY%"=="1" (
    echo Running LateFusion CNN binary for %TAG%
    "%PYTHON%" universe\train\cnn_eda_bvp_latefusion_loso.py ^
        --binary ^
        --eda-dir "%EDA_DIR%" ^
        --bvp-dir "%BVP_DIR%" ^
        --epochs %EPOCHS% ^
        --lr %LR% ^
        --batch-size %BATCH_SIZE% ^
        --dropout %DROPOUT% ^
        --out-dir "%TAG_OUT%\latefusion_cnn" ^
        > "%LOG_DIR%\%TAG%_latefusion_cnn_binary.log" 2>&1
    if errorlevel 1 exit /b 1

    echo Running LateFusion ResNet binary for %TAG%
    "%PYTHON%" universe\train\cnn_eda_bvp_latefusion_resnet_loso.py ^
        --binary ^
        --eda-dir "%EDA_DIR%" ^
        --bvp-dir "%BVP_DIR%" ^
        --epochs %EPOCHS% ^
        --lr %LR% ^
        --batch-size %BATCH_SIZE% ^
        --dropout %DROPOUT% ^
        --out-dir "%TAG_OUT%\latefusion_resnet" ^
        > "%LOG_DIR%\%TAG%_latefusion_resnet_binary.log" 2>&1
    if errorlevel 1 exit /b 1
)

if "%RUN_3CLASS%"=="1" (
    echo Running LateFusion CNN 3-class for %TAG%
    "%PYTHON%" universe\train\cnn_eda_bvp_latefusion_loso.py ^
        --eda-dir "%EDA_DIR%" ^
        --bvp-dir "%BVP_DIR%" ^
        --epochs %EPOCHS% ^
        --lr %LR% ^
        --batch-size %BATCH_SIZE% ^
        --dropout %DROPOUT% ^
        --out-dir "%TAG_OUT%\latefusion_cnn" ^
        > "%LOG_DIR%\%TAG%_latefusion_cnn_3class.log" 2>&1
    if errorlevel 1 exit /b 1

    echo Running LateFusion ResNet 3-class for %TAG%
    "%PYTHON%" universe\train\cnn_eda_bvp_latefusion_resnet_loso.py ^
        --eda-dir "%EDA_DIR%" ^
        --bvp-dir "%BVP_DIR%" ^
        --epochs %EPOCHS% ^
        --lr %LR% ^
        --batch-size %BATCH_SIZE% ^
        --dropout %DROPOUT% ^
        --out-dir "%TAG_OUT%\latefusion_resnet" ^
        > "%LOG_DIR%\%TAG%_latefusion_resnet_3class.log" 2>&1
    if errorlevel 1 exit /b 1
)

exit /b 0

:failed
echo.
echo Window ablation failed. Check logs in %LOG_DIR%.
exit /b 1
