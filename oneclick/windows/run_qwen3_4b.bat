@echo off
rem Pingala-8 one-click: downloads Qwen3-4B B4 (about 1.2 GB) once, then opens a chat page in your browser.
cd /d "%~dp0"
set R=https://huggingface.co/Genrius-AI/Qwen3-4B-Pingala8-2bit/resolve/main
if not exist models mkdir models
for %%F in (4B-csq-vr3-00001-of-00002.gguf 4B-csq-vr3-00002-of-00002.gguf) do (
  if not exist "models\%%F" ( echo Downloading %%F ... & curl.exe -L --fail -o "models\%%F" "%R%/%%F" || goto :fail )
)
call "%~dp0env.bat"
start "" http://127.0.0.1:8080
llama-server.exe -m models\4B-csq-vr3-00001-of-00002.gguf -ngl 99 -c 4096 --port 8080
goto :eof
:fail
echo Download failed. Check your internet connection and run this file again.
pause
