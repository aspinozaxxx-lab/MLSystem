@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0..\..\.."

set "TEST_DIR=tests\handtests\split_train_val_test"
set "INPUT_DIR=%TEST_DIR%\input"
set "OUTPUT_DIR=%TEST_DIR%\output"
set "SCENE_LIST=%INPUT_DIR%\deforestation.txt"
if "%IMAGES_DIR%"=="" set "IMAGES_DIR=E:\Projects\NSPD\Images\Kanopus"

if not exist "%SCENE_LIST%" (
  echo Missing scene list: %SCENE_LIST%
  echo Run %TEST_DIR%\prepare_input.bat first.
  exit /b 1
)

if not exist "%IMAGES_DIR%" (
  echo Missing images directory: %IMAGES_DIR%
  exit /b 1
)

set "ANNOTATION="
set /a GEOJSON_COUNT=0
for %%F in ("%INPUT_DIR%\*.geojson") do (
  if exist "%%~fF" (
    set /a GEOJSON_COUNT+=1
    set "ANNOTATION=%%~fF"
  )
)

if %GEOJSON_COUNT% EQU 0 (
  echo No GeoJSON annotation found in %INPUT_DIR%
  exit /b 1
)

if %GEOJSON_COUNT% GTR 1 (
  echo More than one GeoJSON annotation found in %INPUT_DIR%
  echo Pass --annotation manually or leave exactly one *.geojson file.
  exit /b 1
)

if exist "%OUTPUT_DIR%" rmdir /s /q "%OUTPUT_DIR%"
mkdir "%OUTPUT_DIR%"

python -m mlsystem.src.data.dataset_split split ^
  --scene-list "%SCENE_LIST%" ^
  --images-dir "%IMAGES_DIR%" ^
  --annotation "%ANNOTATION%" ^
  --output-dir "%OUTPUT_DIR%" ^
  --target-val-fraction 0.2 ^
  --seed 42 ^
  --count-mode auto

if errorlevel 1 exit /b %errorlevel%

echo.
echo Main reports:
echo %OUTPUT_DIR%\scene_object_counts.txt
echo %OUTPUT_DIR%\train_val_split.txt
