@echo off
setlocal EnableExtensions

cd /d "%~dp0\..\.."

set "PYTHON=D:\anaconda3\envs\proj\python.exe"
set "EDA_DIR=data\UNIVERSE\windows_eda_30s15s"
set "BVP_DIR=data\UNIVERSE\windows_bvp_30s15s"
set "OUT_ROOT=results\cnn_resnet_binary_optimization"
set "LOG_DIR=%OUT_ROOT%\logs"

set "EPOCHS=40"
set "BATCH_SIZE=64"
set "SEED=0"

if not exist "%PYTHON%" (
    echo Python executable not found: %PYTHON%
    exit /b 1
)

if not exist "%EDA_DIR%\X.npy" (
    echo Missing EDA windows: %EDA_DIR%\X.npy
    echo Run universe\train\run_window_ablation_latefusion.bat first.
    exit /b 1
)

if not exist "%BVP_DIR%\X.npy" (
    echo Missing BVP windows: %BVP_DIR%\X.npy
    echo Run universe\train\run_window_ablation_latefusion.bat first.
    exit /b 1
)

if not exist "%OUT_ROOT%" mkdir "%OUT_ROOT%"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo ============================================================
echo Optimizing one model:
echo   30s/15s LateFusion ResNet binary
echo Results: %OUT_ROOT%
echo Logs:    %LOG_DIR%
echo ============================================================

call :run_config baseline 0.001 0.25 0.0001 ce 0 0 0 0
if errorlevel 1 goto failed

call :run_config lr_3e4 0.0003 0.25 0.0001 ce 1 8 0 0
if errorlevel 1 goto failed

call :run_config dropout_035 0.0003 0.35 0.0001 ce 1 8 0 0
if errorlevel 1 goto failed

call :run_config focal_g1 0.0003 0.25 0.0001 focal 1 8 0 0
if errorlevel 1 goto failed

call :run_config augment 0.0003 0.25 0.0001 ce 1 8 0 1
if errorlevel 1 goto failed

call :run_config balanced_sampler 0.0003 0.25 0.0001 ce 1 8 1 0
if errorlevel 1 goto failed

call :run_config final_combo 0.0003 0.35 0.0001 focal 1 8 1 1
if errorlevel 1 goto failed

echo.
echo Collecting optimization summary
"%PYTHON%" universe\visual\collect_resnet_binary_optimization_results.py ^
    --root "%OUT_ROOT%" ^
    --out-dir "%OUT_ROOT%" ^
    > "%LOG_DIR%\collect_summary.log" 2>&1
if errorlevel 1 goto failed

echo.
echo ============================================================
echo Optimization sweep finished.
echo Summary:
echo   %OUT_ROOT%\summary_all.csv
echo Per-subject:
echo   %OUT_ROOT%\per_subject_all.csv
echo ============================================================
exit /b 0

:run_config
set "CONFIG=%~1"
set "LR=%~2"
set "DROPOUT=%~3"
set "WEIGHT_DECAY=%~4"
set "LOSS=%~5"
set "USE_SCHED=%~6"
set "EARLY_STOP=%~7"
set "BALANCED=%~8"
set "AUGMENT=%~9"
set "FOCAL_GAMMA=1.0"

set "RUN_DIR=%OUT_ROOT%\%CONFIG%\seed_%SEED%"
set "LOG_FILE=%LOG_DIR%\%CONFIG%_seed_%SEED%.log"

echo.
echo ------------------------------------------------------------
echo Config: %CONFIG%
echo lr=%LR% dropout=%DROPOUT% wd=%WEIGHT_DECAY% loss=%LOSS% scheduler=%USE_SCHED% early_stop=%EARLY_STOP% balanced=%BALANCED% augment=%AUGMENT%
echo ------------------------------------------------------------

set "EXTRA_ARGS="
if "%USE_SCHED%"=="1" set "EXTRA_ARGS=%EXTRA_ARGS% --scheduler"
if "%BALANCED%"=="1" set "EXTRA_ARGS=%EXTRA_ARGS% --balanced-sampler"
if "%AUGMENT%"=="1" set "EXTRA_ARGS=%EXTRA_ARGS% --augment"

"%PYTHON%" universe\train\cnn_eda_bvp_latefusion_resnet_binary_optimized.py ^
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
    --focal-gamma %FOCAL_GAMMA% ^
    --early-stop-patience %EARLY_STOP% ^
    %EXTRA_ARGS% ^
    > "%LOG_FILE%" 2>&1
if errorlevel 1 exit /b 1

exit /b 0

:failed
echo.
echo Optimization sweep failed. Check logs in %LOG_DIR%.
exit /b 1
