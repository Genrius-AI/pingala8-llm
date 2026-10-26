@echo off
rem Pingala-8 one-click: downloads Qwen3.5-4B Pingala-8 (about 1.5 GB) once, then opens a chat page in your browser.
cd /d "%~dp0"
set R=https://huggingface.co/Genrius-AI/Qwen3.5-4B-Pingala8-2bit/resolve/main
if not exist models mkdir models
rem Each model part: name, bytes, sha256 (values from the Hugging Face repo).
call :fetch Qwen3.5-4B-Pingala8-2bit-00001-of-00002.gguf 985477344 2d78d7d4eccd4da1a2777ccc26ae399d262205b3fe5468fd30feb12105291050 || goto :fail
call :fetch Qwen3.5-4B-Pingala8-2bit-00002-of-00002.gguf 526138464 caacb42845e6ac3da4bdecec8ae92d157308b56d628eb6feee32a1cca33b1810 || goto :fail
call "%~dp0env.bat"
start "" http://127.0.0.1:8080
llama-server.exe -m models\Qwen3.5-4B-Pingala8-2bit-00001-of-00002.gguf -ngl 99 -c 4096 --port 8080
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
