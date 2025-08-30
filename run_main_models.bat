@echo off
echo Starting batch experiments...

set experiment_name=nothink
set method=agent
set models="mistralai/mistral-7b-instruct-v0.3" "mistralai/mixtral-8x7b-instruct-v0.1" "mistralai/mistral-nemotron" "nv-mistralai/mistral-nemo-12b-instruct" "nvidia/nvidia-nemotron-nano-9b-v2" "nvidia/llama-3.1-nemotron-nano-4b-v1.1" "nvidia/llama-3.1-nemotron-nano-8b-v1" "nvidia/llama-3.3-nemotron-super-49b-v1.5" "nvidia/llama-3.1-nemotron-ultra-253b-v1"


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