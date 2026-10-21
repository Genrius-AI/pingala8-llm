@echo off
rem Pingala-8 one-click: downloads Llama-3.1-8B B4 (about 2.4 GB) once, then opens a chat page in your browser.
rem Built with Llama. Use is subject to the Llama 3.1 Community License and Acceptable Use Policy.
cd /d "%~dp0"
set R=https://huggingface.co/Genrius-AI/Llama-3.1-8B-Pingala8-2bit/resolve/main
if not exist models mkdir models
for %%F in (l31-8b-pingala8-flat-vrnative-cslfull-00001-of-00003.gguf l31-8b-pingala8-flat-vrnative-cslfull-00002-of-00003.gguf l31-8b-pingala8-flat-vrnative-cslfull-00003-of-00003.gguf) do (
  if not exist "models\%%F" ( echo Downloading %%F ... & curl.exe -L --fail -o "models\%%F" "%R%/%%F" || goto :fail )
)
call "%~dp0env.bat"
start "" http://127.0.0.1:8080
llama-server.exe -m models\l31-8b-pingala8-flat-vrnative-cslfull-00001-of-00003.gguf -ngl 99 -c 4096 --port 8080
goto :eof
:fail
echo Download failed. Check your internet connection and run this file again.
pause
