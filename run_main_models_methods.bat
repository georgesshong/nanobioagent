@echo off
echo Starting batch experiments...

set experiment_name=nba
set models="qwen/qwen2.5-coder-7b-instruct" 
set methods=agent

REM advanced version below: task_idx restrict to only specific tasks, config_data drives overwrites (e.g. only do task_classification)
REM set task_idx="0,1"
REM set config_data=data/request_data.json
REM set methods=agent genegpt direct code retrieve
REM set models="nvidia/nvidia-nemotron-nano-9b-v2" "qwen/qwen2.5-coder-7b-instruct" "mistralai/mistral-7b-instruct-v0.3"

for %%m in (%models%) do (
    for %%x in (%methods%) do (
        echo.
        echo ============================================
        echo %date% %time%: Starting %%x with model: %%m
        echo ============================================
        python main.py %experiment_name% --model %%m --method %%x

	REM python main.py %experiment_name% --model %%m --method %%x --task_idx %task_idx% --config_data %config_data% 
        echo %date% %time%: Completed %%x: %%m
    )
)

echo.
echo All batch experiments completed!
