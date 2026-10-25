@echo off
rem Pingala-8 one-click: downloads Llama-3.1-8B B4 (about 2.4 GB) once, then opens a chat page in your browser.
rem Built with Llama. Use is subject to the Llama 3.1 Community License and Acceptable Use Policy.
cd /d "%~dp0"
set R=https://huggingface.co/Genrius-AI/Llama-3.1-8B-Pingala8-2bit/resolve/main
if not exist models mkdir models
rem Each model part: name, bytes, sha256 (values from the Hugging Face repo).
call :fetch l31-8b-pingala8-flat-vrnative-cslfull-00001-of-00003.gguf 994298432 76b39997009a52e35cb31b32a42ad59ec7859c88ccb997d0ae76ed1da4e44686 || goto :fail
call :fetch l31-8b-pingala8-flat-vrnative-cslfull-00002-of-00003.gguf 991815584 fbf44b6f9eac230ff654580592e8881da3dffb78ed98e78c42e1704e15fd2cf0 || goto :fail
call :fetch l31-8b-pingala8-flat-vrnative-cslfull-00003-of-00003.gguf 402096800 8c0146744bc4a91ff067e2cc2fbb2f90fbd0ebb8ee7d279863906a22bc8e67bd || goto :fail
call "%~dp0env.bat"
start "" http://127.0.0.1:8080
llama-server.exe -m models\l31-8b-pingala8-flat-vrnative-cslfull-00001-of-00003.gguf -ngl 99 -c 4096 --port 8080
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
