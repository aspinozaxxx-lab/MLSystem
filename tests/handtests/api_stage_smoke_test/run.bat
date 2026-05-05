@echo off
setlocal

set "TEST_DIR=%~dp0"
pushd "%TEST_DIR%\..\..\.." || exit /b 1
set "PYTHONPATH=%CD%;%PYTHONPATH%"

python tests\handtests\api_stage_smoke_test\run_api_stage_smoke.py
set "RC=%ERRORLEVEL%"

popd
exit /b %RC%
