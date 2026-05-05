@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0..\..\.."

set "SOURCE_INPUT=tests\handtests\input"
set "SOURCE_LIST=%SOURCE_INPUT%\deforestation.txt"
set "TARGET_INPUT=tests\handtests\split_train_val_test\input"
if "%IMAGES_DIR%"=="" set "IMAGES_DIR=E:\Projects\NSPD\Images\Kanopus"
set "REPORT=%SOURCE_INPUT%\deforestation.missing_removed.txt"

if not exist "%SOURCE_INPUT%" (
  echo Missing source handtest input directory: %SOURCE_INPUT%
  exit /b 1
)

if not exist "%SOURCE_LIST%" (
  echo Missing source scene list: %SOURCE_LIST%
  echo Put deforestation.txt into %SOURCE_INPUT% or use the split CLI with --scene-list explicitly.
  exit /b 1
)

if not exist "%IMAGES_DIR%" (
  echo Missing images directory: %IMAGES_DIR%
  exit /b 1
)

mkdir "%TARGET_INPUT%" 2>nul

python -m mlsystem.src.data.dataset_split clean-list ^
  --scene-list "%SOURCE_LIST%" ^
  --images-dir "%IMAGES_DIR%" ^
  --output-report "%REPORT%" ^
  --in-place ^
  --backup

if errorlevel 1 exit /b %errorlevel%

copy /Y "%SOURCE_LIST%" "%TARGET_INPUT%\deforestation.txt" >nul

set "GEOJSON="
set /a GEOJSON_COUNT=0
for %%F in ("%SOURCE_INPUT%\*.geojson") do (
  if exist "%%~fF" (
    set /a GEOJSON_COUNT+=1
    set "GEOJSON=%%~fF"
  )
)

if %GEOJSON_COUNT% EQU 1 (
  copy /Y "%GEOJSON%" "%TARGET_INPUT%\" >nul
) else if %GEOJSON_COUNT% EQU 0 (
  echo No GeoJSON found in %SOURCE_INPUT%; copy annotation manually before run.bat.
) else (
  echo More than one GeoJSON found in %SOURCE_INPUT%; leave exactly one or copy the desired annotation manually.
  exit /b 1
)

echo Prepared input:
echo %TARGET_INPUT%\deforestation.txt
if %GEOJSON_COUNT% EQU 1 for %%A in ("%GEOJSON%") do echo %TARGET_INPUT%\%%~nxA
echo Report:
echo %REPORT%
