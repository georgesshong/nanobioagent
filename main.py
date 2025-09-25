#!/usr/bin/env python3
'''
NBA: Nano Bio-Agent
    - S/LLM agents use API (eg NCBI, AlphaGenome, etc) to answer questions about genes
    - support various LLM models via LangChain, Ollama and HuggingFace
    - support various methods: api call, direct llm call...
    - support caching of api calls for performance
'''
import io
import os
import sys
import json
import time
import tqdm
import shutil
import hashlib
import argparse
import datetime
from tqdm import tqdm
from contextlib import contextmanager
from dotenv import load_dotenv
load_dotenv()

from nanobioagent.core.prompts import get_prompt_header
from nanobioagent.api import gene_answer
from nanobioagent.tools.gene_utils import log_event  # Make sure this import exists
from nanobioagent.evaluation.evaluate import main_evaluation, TASK_TYPE_MAPPING
from nanobioagent.evaluation.compare import calculate_score
try:
    import nanobioagent as nba
    PACKAGE_MODE = True
    log_event("Running in package mode")
except ImportError as e:
    PACKAGE_MODE = False
    log_event(f"Package import failed, using local imports: {e}")

try:
    from langchain_community.llms import OpenAI, HuggingFacePipeline
    from langchain_community.chat_models import ChatOpenAI, ChatGooglePalm, ChatOllama
    from langchain_anthropic import ChatAnthropic
    from langchain.schema import HumanMessage, SystemMessage
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    import torch
    LANGCHAIN_AVAILABLE = True
except ImportError:
    print("LangChain not found. Please install it with: pip install langchain langchain-openai langchain-anthropic langchain-community")
    print("Falling back to OpenAI only.")
    import openai
    LANGCHAIN_AVAILABLE = False
    openai.api_key = os.environ.get('OPENAI_API_KEY')

SKIP_REPEAT_PROMPT = True
MAX_NUM_CALLS = 10
DEFAULT_MODEL_NAME = "gpt-4o-mini" # Default model for LLM calls in this module
DEFAULT_RESULTS_FOLDER = "default"
DEFAULT_STR_MASK = "111111"

@contextmanager
def suppress_stdout():
    """Context manager to temporarily suppress stdout output."""
    original_stdout = sys.stdout
    sys.stdout = io.StringIO()  # Redirect stdout to a string buffer
    try:
        yield
    finally:
        sys.stdout = original_stdout  # Restore original stdout

# --task_idx
# 0: "Gene alias"
# 1: "Gene disease association"
# 2: "Gene location"
# 3: "Human genome DNA aligment"
# 4: "Multi-species DNA aligment"
# 5: "Gene name conversion"
# 6: "Protein-coding genes"
# 7: "Gene SNP association"
# 8: "SNP location"

def parse_arguments():
    """Parse command line arguments and return the args object."""
    parser = argparse.ArgumentParser(description='NanoBioAgent: Multi-agent genomics framework')
    # Make str_mask optional with nargs='?'
    parser.add_argument('experiment_name', nargs='?', default='',
            help='Experiment name for output folder. If 6-digit binary (e.g., 111111), '
                'used as str_mask for ablation studies. If empty, defaults to "default".')
    parser.add_argument('-m', '--model', type=str, default='gpt-3.5-turbo', 
            help='Model name to use (default: gpt-3.5-turbo)')
    parser.add_argument('--config_data', type=str, default=None,
        help='Path to config file controlling operations and other overwrites')
    parser.add_argument('--local', action='store_true',
            help='Force using local model from HuggingFace')
    parser.add_argument('--hf_path', type=str,
            help='Path to HuggingFace model (for local models)')
    parser.add_argument('--use-fallback', action='store_true', 
            help='Force using direct API calls instead of LangChain')
    parser.add_argument('--task_idx', type=str, 
            help='Comma-separated list of task indices to run (0-indexed, e.g., "0,1,4"). Default: run all tasks')
    parser.add_argument('--method', type=str, default='genegpt',
            help='Method to use: genegpt/g (original), direct/d (model knowledge only), retrieve/r (retrieve from json), code/c (future implementation)')
    parser.add_argument('--qas_file', type=str, default='data/geneturing.json',
            help='Files containing the questions and expected answers (default: data/geneturing.json)')
    
    args = parser.parse_args()
    
    # Normalize method argument to handle aliases
    if args.method.lower() in ['g', 'genegpt']:
        args.method = 'genegpt'
    elif args.method.lower() in ['d', 'direct']:
        args.method = 'direct'
    elif args.method.lower() in ['r', 'retrieve']:
        args.method = 'retrieve'
    elif args.method.lower() in ['c', 'code']:
        args.method = 'code'
    elif args.method.lower() in ['a', 'agent']:
        args.method = 'agent'
    else:
        log_event(f"Warning: Unknown method '{args.method}'. Defaulting to 'agent'.")
        args.method = 'agent'
    return args

def process_experiment_name(experiment_name):
    """
    Process experiment name and determine str_mask and output folder.
    Returns:
        tuple: (output_folder, str_mask, mask_list)
    """
    if not experiment_name:
        # Case 1: Empty - use default
        output_folder = DEFAULT_RESULTS_FOLDER
        str_mask = DEFAULT_STR_MASK
    elif len(experiment_name) == 6 and experiment_name.isdigit() and all(c in '01' for c in experiment_name):
        # Case 2: 6-digit binary string - ablation study
        output_folder = experiment_name
        str_mask = experiment_name
    else:
        # Case 3: Any other string - use as folder name with default mask
        output_folder = experiment_name
        str_mask = DEFAULT_STR_MASK
    # Convert str_mask to boolean mask
    mask = [bool(int(x)) for x in str_mask]
    
    return output_folder, str_mask, mask

# Create timestamped snapshots of multiple files in organized directories.
# for debugging and tracking changes only
def get_files_to_snap():
    return [
            'main.py',  # Current script 
            'nanobioagent/api.py',  # New API module
            'nanobioagent/core/agent_framework.py',
            'nanobioagent/core/model_utils.py',
            'nanobioagent/core/prompts.py',
            'nanobioagent/tools/ncbi_query.py',
            'nanobioagent/tools/gene_utils.py',
            'nanobioagent/tools/ncbi_tools.py',
            'nanobioagent/integration/gene_gpt.py',
            'nanobioagent/evaluation/evaluate.py',
            'nanobioagent/evaluation/compare.py',
            'nanobioagent/config/langchain_config.json',
            'nanobioagent/config/plans.json',
            'nanobioagent/config/tasks.json',
            'nanobioagent/config/examples/ncbi_examples.json',
            'nanobioagent/config/tools/ncbi_tools.json',
            'nanobioagent/__init__.py',
            'requirements.txt'
        ]
    
def save_timestamped_copy(files_to_snap=None):
    # Default files to snapshot (relative to main.py location)
    if files_to_snap is None:
        files_to_snap = get_files_to_snap()
    
    # Get the directory where main.py is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Put archives inside results directory
    results_dir = os.path.join(script_dir, 'results')
    archive_dir = os.path.join(results_dir, 'archives')  # Changed this line
    os.makedirs(archive_dir, exist_ok=True)  # Use makedirs instead of mkdir
    
    # Calculate combined hash of all files
    combined_hash = calculate_combined_hash(script_dir, files_to_snap)
    
    # Check if any changes occurred since last snapshot
    if not has_changes_since_last_snapshot(archive_dir, combined_hash):
        log_event("No changes detected in tracked files. Skipping snapshot.")
        return
    
    # Create new timestamped snapshot
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_dir = os.path.join(archive_dir, timestamp)
    os.makedirs(snapshot_dir, exist_ok=True)
    
    # Copy files maintaining directory structure
    copied_files = []
    skipped_files = []
    
    for file_path in files_to_snap:
        source_path = os.path.join(script_dir, file_path)
        
        if not os.path.exists(source_path):
            skipped_files.append(file_path)
            continue
        
        # Create destination path maintaining structure
        dest_path = os.path.join(snapshot_dir, file_path)
        dest_dir = os.path.dirname(dest_path)
        
        # Create directory structure if needed
        if dest_dir:
            os.makedirs(dest_dir, exist_ok=True)
        
        # Copy file
        shutil.copy2(source_path, dest_path)
        copied_files.append(file_path)
    
    # Save metadata about this snapshot
    save_snapshot_metadata(snapshot_dir, timestamp, combined_hash, copied_files, skipped_files)
    
    log_event(f"Created snapshot: archive/{timestamp}/")
    log_event(f"  Copied files: {len(copied_files)}")
    if skipped_files:
        log_event(f"  Skipped files (not found): {skipped_files}")
    
    # Clean up old snapshots (keep last 10)
    cleanup_old_snapshots(archive_dir, keep_count=10)

def calculate_combined_hash(script_dir, files_to_snap):
    """Calculate combined SHA256 hash of all tracked files."""
    hasher = hashlib.sha256()
    
    for file_path in sorted(files_to_snap):  # Sort for consistent hashing
        source_path = os.path.join(script_dir, file_path)
        
        if os.path.exists(source_path):
            with open(source_path, 'rb') as f:
                content = f.read()
            hasher.update(f"{file_path}:".encode())  # Include filename in hash
            hasher.update(content)
        else:
            # Include info about missing files in hash
            hasher.update(f"{file_path}:MISSING".encode())
    
    return hasher.hexdigest()

def has_changes_since_last_snapshot(archive_dir, current_hash):
    """Check if there are changes since the last snapshot."""
    # Get all snapshot directories
    snapshot_dirs = [
        d for d in os.listdir(archive_dir) 
        if os.path.isdir(os.path.join(archive_dir, d)) and d.replace('_', '').isdigit()
    ]
    
    if not snapshot_dirs:
        return True  # No previous snapshots, so this is a change
    
    # Get the latest snapshot
    latest_snapshot = max(snapshot_dirs)
    metadata_file = os.path.join(archive_dir, latest_snapshot, '.snapshot_metadata.json')
    
    if not os.path.exists(metadata_file):
        return True  # No metadata, assume changes
    
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        return metadata.get('combined_hash') != current_hash
    except (json.JSONDecodeError, KeyError):
        return True  # Error reading metadata, assume changes

def save_snapshot_metadata(snapshot_dir, timestamp, combined_hash, copied_files, skipped_files):
    """Save metadata about the snapshot."""
    metadata = {
        'timestamp': timestamp,
        'combined_hash': combined_hash,
        'copied_files': copied_files,
        'skipped_files': skipped_files,
        'created_at': datetime.datetime.now().isoformat()
    }
    
    metadata_file = os.path.join(snapshot_dir, '.snapshot_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

def cleanup_old_snapshots(archive_dir, keep_count=30):
    """Remove old snapshots, keeping only the most recent ones."""
    snapshot_dirs = [
        d for d in os.listdir(archive_dir) 
        if os.path.isdir(os.path.join(archive_dir, d)) and d.replace('_', '').isdigit()
    ]
    
    if len(snapshot_dirs) <= keep_count:
        return  # Nothing to clean up
    
    # Sort by timestamp (directory name) and remove oldest
    sorted_snapshots = sorted(snapshot_dirs)
    to_remove = sorted_snapshots[:-keep_count]
    
    for snapshot_dir in to_remove:
        snapshot_path = os.path.join(archive_dir, snapshot_dir)
        shutil.rmtree(snapshot_path)
        log_event(f"Removed old snapshot: {snapshot_dir}")

# helper function to set high priority on Windows
def set_high_priority():
    if os.name != 'nt':
        return
    try:
        import subprocess
        subprocess.run(['wmic', 'process', 'where', f'processid={os.getpid()}', 'CALL', 'setpriority', 'high priority'], capture_output=True, shell=True, check=True)
        print("Process priority set to high!")
    except:
        print("Process priority could not be set!")
        pass  # Silently fail on non-Windows or if wmic fails

# helper function to control Windows sleep/display timeout behavior, i needed this for long runs on my laptop
def control_sleep_windows(state=None):
    # set state to "prevent" to disable sleep/display timeout, "allow" to restore normal sleep behavior, None to do nothing
    if state is None:
        return
    # Check if running on Windows
    if os.name != 'nt':
        return  # Not Windows, skip
    try:
        import ctypes
        if state == "prevent":
            # ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000002)
        elif state == "allow":
            # ES_CONTINUOUS - restore normal power management
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    except (AttributeError, ImportError, OSError):
        # Handle cases where ctypes or windll don't exist, or call fails
        pass

# Main function to run the analysis
def main():
    # Set high priority for the process
    set_high_priority()
    # Snap a copy of the current script with a timestamp
    save_timestamped_copy()
    # Prevent sleep & display turning off for windows
    control_sleep_windows(state="prevent")
    # Parse command line arguments
    args = parse_arguments()
    
    # Process experiment name and determine output structure
    output_folder, str_mask, mask = process_experiment_name(args.experiment_name)
    
    model_name = args.model
    method = args.method
    qas_file = args.qas_file
    
    # If using local model with specific path
    if args.local and args.hf_path:
        model_name = args.hf_path
    
    # Load configuration if available
    config_data = None
    if args.config_data:
        with open(args.config_data, 'r') as f:
            config_data = json.load(f)
            
    log_event(f"Experiment: {output_folder}")
    log_event(f"Str_mask: {str_mask} (mask: {mask})")
    log_event(f"Using model: {model_name}, Method: {method}")
    
    # Only generate prompt if using the 'genegpt' method (needs the mask)
    prompt = None if method != 'genegpt' else get_prompt_header(mask)

    # Create organized results directory structure
    results_base_dir = "results"
    experiment_dir = os.path.join(results_base_dir, output_folder)
    
    # Clean up model name for directory
    model_dir_name = model_name.replace("/", "_").replace(":", "_")
    if args.use_fallback:
        model_dir_name += "_fallback"
    model_dir_name += f"_{method}"
    
    request_type = "answer" # Default request type
    if config_data and config_data.get("request", {}).get("type"):
        request_type = config_data["request"]["type"] # overwrite if specified in config
        result_dir = os.path.join(experiment_dir, model_dir_name, request_type)
    else:
        result_dir = os.path.join(experiment_dir, model_dir_name)

    os.makedirs(result_dir, exist_ok=True)
    
    log_event(f"Results will be saved to: {result_dir}")
    # initialize 
    prev_call = time.time()	
    qas = json.load(open(qas_file))
    
    # Process task_idx argument - parse the comma-separated list of task indices (0-indexed)
    task_indices = None
    if args.task_idx:
        try:
            # Parse as 0-indexed list
            task_indices = [int(idx.strip()) for idx in args.task_idx.split(',')]
            # Get task names by index
            all_task_names = list(qas.keys())
            filtered_tasks = {}
            
            for idx in task_indices:
                if 0 <= idx < len(all_task_names):
                    task_name = all_task_names[idx]
                    filtered_tasks[task_name] = qas[task_name]
                else:
                    log_event(f"Warning: Task index {idx} is out of range (valid range: 0-{len(all_task_names)-1})")
            
            if filtered_tasks:
                qas = filtered_tasks
                log_event(f"Running only tasks: {', '.join([all_task_names[idx] for idx in task_indices if 0 <= idx < len(all_task_names)])}")
            else:
                log_event("No valid task indices provided. Using all tasks.")
        except ValueError:
            log_event(f"Invalid task index format: {args.task_idx}. Using all tasks.")
    
    # Create a master progress bar for all tasks
    total_questions = sum(len(info) for info in qas.values())
    master_pbar = tqdm(total=total_questions, desc="Overall Progress", position=0)
    
    # Track completed questions to update the master progress bar
    total_completed = 0
    
    # Get the full list of tasks for showing correct task indices
    all_task_names = list(json.load(open(qas_file)).keys())
    
    for task, info in qas.items():
        try:
            # Calculate the 0-indexed task index in the full task list
            task_idx = all_task_names.index(task)
            
            if os.path.exists(os.path.join(result_dir, f'{task}.json')):
                # continue if task is done
                preds = json.load(open(os.path.join(result_dir, f'{task}.json')))
                if len(preds) == 50:
                    # Update master progress bar for completed tasks
                    master_pbar.update(len(info))
                    total_completed += len(info)
                    continue
                output = preds
            else:
                output = []
            
            done_questions = set([entry[0] for entry in output])
            # Update master progress bar for questions already completed in this task
            master_pbar.update(len(done_questions))
            total_completed += len(done_questions)
            
            log_event(f'\nTask {task_idx}/8: {task}\n')  # Using 0-indexed display
            # Create a progress bar for this specific task
            task_total = len(info)
            task_completed = len(done_questions)
            task_pbar = tqdm(total=task_total, desc=f"Task: {task}", position=1, leave=False)
            task_pbar.update(task_completed)  # Update with already completed questions
            
            expected_task_type_for_classification = None
            if request_type == "task_classification":
                expected_task_types = TASK_TYPE_MAPPING.get(task, [])
                if expected_task_types:
                    expected_task_type_for_classification = expected_task_types[0] if len(expected_task_types) == 1 else ','.join(expected_task_types)
                else:
                    expected_task_type_for_classification = "unknown"
                    
            for question, answer in info.items():
                try:
                    if question in done_questions:
                        continue
                        
                    log_event('\n---New Instance---')
                    expected_answer = answer
                    if request_type == "task_classification":
                        expected_answer = expected_task_type_for_classification
                        
                    # Process the question using the method parameter
                    result = gene_answer(
                        question=question,
                        answer=expected_answer,
                        prompt=prompt,
                        model_name=model_name,
                        config_data=config_data,
                        use_fallback=args.use_fallback,
                        method=method
                    )
                    
                    # Add result to output
                    output.append(result)
                    
                    # After processing the question (successful or error)
                    task_pbar.update(1)  # Update task progress bar
                    master_pbar.update(1)  # Update master progress bar
                    total_completed += 1
                    
                    # Save results after each question
                    with open(os.path.join(result_dir, f'{task}.json'), 'w') as f:
                        json.dump(output, f, indent=4)
                except Exception as e:
                    log_event(f"Error processing question: {str(e)}")
                    # Try to save what we have so far
                    with open(os.path.join(result_dir, f'{task}.json'), 'w') as f:
                        json.dump(output, f, indent=4)
                    
            # Close the task progress bar when done
            task_pbar.close()
        except Exception as e:
            log_event(f"Error processing task '{task}': {str(e)}")
            continue
    
    # Close the master progress bar
    master_pbar.close()
    
    # Print a final summary
    log_event(f"\nProcessing complete: {total_completed}/{total_questions} questions processed")
    # Allow sleep again
    control_sleep_windows(state="allow")

    # Auto-run evaluation
    try:
        print("\n" + "="*50)
        print("Auto-running Evaluation")
        print("="*50)
        
        main_evaluation(result_dir, qas_file=qas_file)
        
        log_event(f"\nEvaluation complete")
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

def cli_main():
    """
    Simple CLI entry point for package installation.
    This is different from main() which runs the full evaluation pipeline.
    """
    if len(sys.argv) < 2:
        print("NanoBioAgent CLI Usage:")
        print("  nanobioagent 'Your biological question here'")
        print("  Example: nanobioagent 'What is the official gene symbol of BRCA1?'")
        print("")
        print("For full evaluation pipeline, use: python main.py [args]")
        return
    
    question = " ".join(sys.argv[1:])
    
    try:
        if PACKAGE_MODE:
            result = nba.answer(question)
            print(f"Question: {question}")
            print(f"Answer: {result}")
        else:
            # Fallback: use the gene_answer function directly
            result = gene_answer(question, method='agent', model_name=DEFAULT_MODEL_NAME)
            answer = result[2] if isinstance(result, list) and len(result) >= 3 else str(result)
            print(f"Question: {question}")
            print(f"Answer: {answer}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Check if this is being called as a simple CLI (from package installation)
    # vs the full evaluation pipeline
    
    # If called with simple question (no special args), use CLI mode
    if len(sys.argv) >= 2 and not any(arg.startswith('-') for arg in sys.argv[1:]):
        # Simple question mode - use CLI
        cli_main()
    else:
        # Full evaluation pipeline mode - use original main
        main()