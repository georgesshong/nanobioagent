@echo off
echo Starting batch experiments...

set experiment_name=nothink
REM set models="qwen/qwen3-235b-a22b" "qwen/qwen2.5-coder-32b-instruct" "qwen/qwen2.5-coder-7b-instruct"
set models="gpt-4.1" "gpt-4.1-mini" "gpt-4.1-nano" "gemini-1.5-flash" "gemini-2.0-flash"

set method=genegpt

for %%m in (%models%) do (
    echo.
    echo ============================================
    echo %date% %time%: Starting experiment with model: %%m
    echo ============================================
    python main.py %experiment_name% --model %%m --method %method%
    echo %date% %time%: Completed: %%m
)

set method=direct
for %%m in (%models%) do (
    echo.
    echo ============================================
    echo %date% %time%: Starting experiment with model: %%m
    echo ============================================
    python main.py %experiment_name% --model %%m --method %method%
    echo %date% %time%: Completed: %%m
)

echo.
echo All batch experiments completed!
pause