@echo off
setlocal enabledelayedexpansion
echo ============================================
echo %date% %time%: Testing enhanced get_answer on filtered folders
echo ============================================

set dir_results=results\nba_finals

REM Filter 1: Include only folders containing specific text (leave empty to include all)
REM Examples: "gpt", "claude", "gemini", "qwen"
set include_filter=

REM Filter 2: Exclude folders containing specific text (leave empty to exclude none)
REM Examples: "test", "backup", "old", "evaluation_reports"
set exclude_filter=evaluation_reports zzz

REM Filter 3: Limit to first N folders (set to 0 for no limit)
set max_folders=0

if not "%include_filter%"=="" (
    echo Include filter: Only folders containing "%include_filter%"
)
if not "%exclude_filter%"=="" (
    echo Exclude filter: Skipping folders containing "%exclude_filter%"
)
if not "%max_folders%"=="0" (
    echo Folder limit: Testing maximum %max_folders% folders
)

echo Scanning folders in %dir_results%...
echo.

set folder_count=0

REM Loop through all folders in the results directory
for /d %%F in ("%dir_results%\*") do (
    set folder_name=%%~nxF
    set skip_folder=0
    
    REM echo DEBUG: Found folder: !folder_name!
    
    REM Check include filter
    if not "%include_filter%"=="" (
        echo !folder_name! | findstr /i "%include_filter%" >nul
        if errorlevel 1 (
            echo DEBUG: !folder_name! does not contain "%include_filter%"
            set skip_folder=1
        ) else (
            echo DEBUG: !folder_name! matches include filter "%include_filter%"
        )
    )
    
    REM Check exclude filter
    if not "%exclude_filter%"=="" (
        echo !folder_name! | findstr /i "%exclude_filter%" >nul
        if not errorlevel 1 (
            echo DEBUG: !folder_name! matches exclude filter "%exclude_filter%"
            set skip_folder=1
        )
    )
    
    REM Check folder limit
    if not "%max_folders%"=="0" (
        if !folder_count! geq %max_folders% (
            echo DEBUG: Reached folder limit %max_folders%
            set skip_folder=1
        )
    )
    
    echo DEBUG: skip_folder=!skip_folder! for !folder_name!
    
    if !skip_folder!==0 (
        set /a folder_count+=1
        echo ============================================
        echo Testing folder !folder_count!: !folder_name!
        echo ============================================
        
        REM Run evaluation on each folder to test enhanced get_answer function
        REM python -m nanobioagent.evaluation.evaluate "%%F"
        python nanobioagent/evaluation/evaluate.py "%%F"

        echo.
        echo Test completed for !folder_name!
        echo.
        echo ============================================
        echo.
    ) else (
        echo Skipping folder: !folder_name! (filtered out)
    )
)

echo.
echo Tested %folder_count% folders total
echo All folder tests completed!