@echo off
setlocal EnableExtensions

cd /d "%~dp0\..\.."

set "PYTHON=D:\anaconda3\envs\proj\python.exe"
set "EDA_DIR=data\UNIVERSE\windows_eda_30s15s"
set "BVP_DIR=data\UNIVERSE\windows_bvp_30s15s"
set "OUT_ROOT=results\cnn_attention_binary"
set "LOG_DIR=%OUT_ROOT%\logs"

set "EPOCHS=40"
set "BATCH_SIZE=64"
set "SEED=0"
set "LR=0.0003"
set "DROPOUT=0.25"
set "WEIGHT_DECAY=0.0001"
set "LOSS=ce"
set "EARLY_STOP=8"

if not exist "%PYTHON%" (
    echo Python executable not found: %PYTHON%
    exit /b 1
)

if not exist "%EDA_DIR%\X.npy" (
    echo Missing EDA windows: %EDA_DIR%\X.npy
    exit /b 1
)

if not exist "%BVP_DIR%\X.npy" (
    echo Missing BVP windows: %BVP_DIR%\X.npy
    exit /b 1
)

if not exist "%OUT_ROOT%" mkdir "%OUT_ROOT%"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo ============================================================
echo Attention pooling extension experiment
echo   30s/15s LateFusion ResNet binary
echo Results: %OUT_ROOT%
echo Logs:    %LOG_DIR%
echo ============================================================

call :run_pool avg avg_pool_baseline
if errorlevel 1 goto failed

call :run_pool attn attention_pool
if errorlevel 1 goto failed

echo.
echo Collecting attention summary
"%PYTHON%" universe\visual\collect_attention_binary_results.py ^
    --root "%OUT_ROOT%" ^
    > "%LOG_DIR%\collect_summary.log" 2>&1
if errorlevel 1 goto failed

echo.
echo ============================================================
echo Attention experiment finished.
echo Summary:
echo   %OUT_ROOT%\summary_all.csv
echo Per-subject:
echo   %OUT_ROOT%\per_subject_all.csv
echo ============================================================
exit /b 0

:run_pool
set "POOL=%~1"
set "CONFIG=%~2"
set "RUN_DIR=%OUT_ROOT%\%CONFIG%\seed_%SEED%"
set "LOG_FILE=%LOG_DIR%\%CONFIG%_seed_%SEED%.log"

echo.
echo ------------------------------------------------------------
echo Config: %CONFIG% pool=%POOL%
echo ------------------------------------------------------------

"%PYTHON%" universe\train\cnn_eda_bvp_latefusion_resnet_attention_binary.py ^
    --eda-dir "%EDA_DIR%" ^
    --bvp-dir "%BVP_DIR%" ^
    --out-dir "%RUN_DIR%" ^
    --config-name "%CONFIG%" ^
    --run-seed %SEED% ^
    --epochs %EPOCHS% ^
    --batch-size %BATCH_SIZE% ^
    --lr %LR% ^
    --dropout %DROPOUT% ^
    --weight-decay %WEIGHT_DECAY% ^
    --loss %LOSS% ^
    --scheduler ^
    --early-stop-patience %EARLY_STOP% ^
    --pool %POOL% ^
    > "%LOG_FILE%" 2>&1
if errorlevel 1 exit /b 1

exit /b 0

:failed
echo.
echo Attention experiment failed. Check logs in %LOG_DIR%.
exit /b 1
