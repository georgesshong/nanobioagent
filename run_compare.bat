@echo off
echo ============================================
echo %date% %time%: Compare results (2-1)
echo ============================================

set experiment_name_1=nothink
set experiment_name_2=nba

REM set model_method_1=qwen_qwen2.5-coder-7b-instruct_agent
REM set model_method_2=qwen_qwen2.5-coder-7b-instruct_agent
set model_method_1=qwen_qwen3-coder-480b-a35b-instruct_agent
set model_method_2=qwen_qwen3-coder-480b-a35b-instruct_agent

python -m nanobioagent.evaluation.compare "results\%experiment_name_1%\%model_method_1%" "results\%experiment_name_2%\%model_method_2%"
REM python nanobioagent/evaluation/compare.py "results\%experiment_name_1%\%model_method_1%" "results\%experiment_name_2%\%model_method_2%"

echo.
echo Compare completed!