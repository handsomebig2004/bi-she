@echo off
setlocal

cd /d "%~dp0\..\.."

set "PYTHON=D:\anaconda3\envs\proj\python.exe"
set "OUT_ROOT=results\cnn_latefusion"
set "LOG_DIR=%OUT_ROOT%\logs"
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
echo Running UNIVERSE late-fusion CNN experiments
echo Output root: %OUT_ROOT%
echo Logs:        %LOG_DIR%
echo ============================================================

echo.
echo [1/4] LateFusion CNN 3-class
"%PYTHON%" universe\train\cnn_eda_bvp_latefusion_loso.py ^
    --epochs %EPOCHS% ^
    --lr %LR% ^
    --batch-size %BATCH_SIZE% ^
    --dropout %DROPOUT% ^
    --out-dir "%OUT_ROOT%\latefusion_cnn" ^
    > "%LOG_DIR%\latefusion_cnn_3class.log" 2>&1
if errorlevel 1 goto failed

echo.
echo [2/4] LateFusion CNN binary
"%PYTHON%" universe\train\cnn_eda_bvp_latefusion_loso.py ^
    --binary ^
    --epochs %EPOCHS% ^
    --lr %LR% ^
    --batch-size %BATCH_SIZE% ^
    --dropout %DROPOUT% ^
    --out-dir "%OUT_ROOT%\latefusion_cnn" ^
    > "%LOG_DIR%\latefusion_cnn_binary.log" 2>&1
if errorlevel 1 goto failed

echo.
echo [3/4] LateFusion ResNet 3-class
"%PYTHON%" universe\train\cnn_eda_bvp_latefusion_resnet_loso.py ^
    --epochs %EPOCHS% ^
    --lr %LR% ^
    --batch-size %BATCH_SIZE% ^
    --dropout %DROPOUT% ^
    --out-dir "%OUT_ROOT%\latefusion_resnet" ^
    > "%LOG_DIR%\latefusion_resnet_3class.log" 2>&1
if errorlevel 1 goto failed

echo.
echo [4/4] LateFusion ResNet binary
"%PYTHON%" universe\train\cnn_eda_bvp_latefusion_resnet_loso.py ^
    --binary ^
    --epochs %EPOCHS% ^
    --lr %LR% ^
    --batch-size %BATCH_SIZE% ^
    --dropout %DROPOUT% ^
    --out-dir "%OUT_ROOT%\latefusion_resnet" ^
    > "%LOG_DIR%\latefusion_resnet_binary.log" 2>&1
if errorlevel 1 goto failed

echo.
echo ============================================================
echo All experiments finished.
echo.
echo Result files:
echo   %OUT_ROOT%\latefusion_cnn\loso_results_kept.csv
echo   %OUT_ROOT%\latefusion_cnn\loso_results_skipped.csv
echo   %OUT_ROOT%\latefusion_cnn\binary_loso\loso_results_kept.csv
echo   %OUT_ROOT%\latefusion_cnn\binary_loso\loso_results_skipped.csv
echo   %OUT_ROOT%\latefusion_resnet\loso_results_kept.csv
echo   %OUT_ROOT%\latefusion_resnet\loso_results_skipped.csv
echo   %OUT_ROOT%\latefusion_resnet\binary_loso\loso_results_kept.csv
echo   %OUT_ROOT%\latefusion_resnet\binary_loso\loso_results_skipped.csv
echo ============================================================
exit /b 0

:failed
echo.
echo Experiment failed. Check logs in %LOG_DIR%.
exit /b 1
