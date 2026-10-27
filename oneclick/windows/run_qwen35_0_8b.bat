@echo off
rem Pingala-8 one-click: downloads Qwen3.5-0.8B Pingala-8 (about 0.35 GB) once, then opens a chat page in your browser.
cd /d "%~dp0"
set R=https://huggingface.co/Genrius-AI/Qwen3.5-0.8B-Pingala8-2bit/resolve/main
if not exist models mkdir models
rem Each model part: name, bytes, sha256 (values from the Hugging Face repo).
call :fetch Qwen3.5-0.8B-Pingala8-2bit.gguf 346833696 8c80d48787d0dcd1a8aafb4d47d3e0e00c73214b103b376bfa764cc4c771df27 || goto :fail
call "%~dp0env.bat"
start "" http://127.0.0.1:8080
rem Print the star line once the server answers (its /health returns 200 after the model has loaded).
start "" /b cmd /c "for /l %%i in (1,1,900) do @(curl.exe -sf -o nul http://127.0.0.1:8080/health && (echo Running? A star helps others find it: https://github.com/Genrius-AI/pingala8-llm& exit) || ping -n 2 127.0.0.1 >nul)"
llama-server.exe -m models\Qwen3.5-0.8B-Pingala8-2bit.gguf -ngl 99 -c 4096 --port 8080
goto :eof
:fail
echo Download failed. Check your internet connection and run this file again.
pause
goto :eof

rem fetch NAME BYTES SHA256: keep a complete file; otherwise download to NAME.part, check it, then rename it.
:fetch
set "PF=models\%~1"
set TRY=0
if not exist "%PF%" goto :fetch_get
for %%A in ("%PF%") do set "HAVE=%%~zA"
if "%HAVE%"=="%~2" exit /b 0
echo %PF% is %HAVE% bytes, expected %~2 (incomplete download). Downloading it again.
del /f /q "%PF%"
:fetch_get
set /a TRY+=1
if exist "%PF%.part" del /f /q "%PF%.part"
echo Downloading %~1 ...
curl.exe -L --fail -o "%PF%.part" "%R%/%~1"
if errorlevel 1 goto :fetch_cut
for %%A in ("%PF%.part") do set "HAVE=%%~zA"
if not "%HAVE%"=="%~2" goto :fetch_size
certutil -hashfile "%PF%.part" SHA256 | findstr /i /x "%~3" >nul
if errorlevel 1 goto :fetch_sum
move /y "%PF%.part" "%PF%" >nul
exit /b 0
:fetch_cut
echo Download of %~1 did not finish.
goto :fetch_bad
:fetch_size
echo Size check failed for %~1: got %HAVE% bytes, expected %~2.
goto :fetch_bad
:fetch_sum
echo Checksum (sha256) check failed for %~1.
:fetch_bad
if exist "%PF%.part" del /f /q "%PF%.part"
if %TRY% geq 3 exit /b 1
set /a NEXT=TRY+1
echo Trying again (%NEXT% of 3) ...
goto :fetch_get
