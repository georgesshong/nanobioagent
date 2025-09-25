'''
Compare GeneGPT results from multiple folders and generate a CSV comparison
Usage: python compare_results.py folder_A folder_B [folder_C] [folder_D] [details_output.csv] [summary_output.csv]
All comparisons are relative to the first folder (folder_A as baseline)
'''

import glob
import json
import os
import sys
import csv
import re
from typing import Dict, List, Tuple, Any

# Import shared functions from evaluate.py to avoid duplication
from .evaluate import (
    extract_chromosome, 
    extract_chromosome_positions,
    split_comma_separated, 
    normalize_gene_symbol, 
    get_answer,
    extract_token_usage_info,
    extract_stages_completed
)

def calculate_score(prediction: Any, ground_truth: Any, task: str) -> float:
    """
    Calculate score for a prediction based on task type (updated to match evaluate.py logic)
    """
    try:
        if task == 'Gene disease association':
            if isinstance(ground_truth, str):
                ground_truth = split_comma_separated(ground_truth)
            if isinstance(prediction, str):
                prediction = split_comma_separated(prediction)
            
            if not ground_truth:
                return 0.0
            
            # Normalize gene symbols for comparison (matching evaluate.py)
            normalized_ground_truth = [normalize_gene_symbol(gene) for gene in ground_truth]
            normalized_prediction = [normalize_gene_symbol(gene) for gene in prediction] if prediction else []
            
            answer_in = [ans in normalized_prediction for ans in normalized_ground_truth]
            return sum(answer_in) / len(answer_in)

        elif task == 'Disease gene location':
            if isinstance(ground_truth, list):
                pass
            elif isinstance(ground_truth, str):
                ground_truth = split_comma_separated(ground_truth)
            
            if isinstance(prediction, str):
                prediction = split_comma_separated(prediction)
                
            if not ground_truth:
                return 0.0
                
            answer_in = [ans in prediction for ans in ground_truth]
            return sum(answer_in) / len(answer_in)

        elif task == 'Human genome DNA aligment':
            if isinstance(prediction, str) and isinstance(ground_truth, str):
                # Use the enhanced position matching from evaluate.py
                pred_chr, pred_start, pred_end = extract_chromosome_positions(prediction)
                answer_chr, answer_start, answer_end = extract_chromosome_positions(ground_truth)
                
                # Exact match gets full score
                if prediction == ground_truth:
                    return 1.0
                # Position match gets 0.75 score (NEW - matching evaluate.py)
                elif pred_chr == answer_chr and pred_chr != "":
                    if (pred_start is not None and pred_end is not None and 
                        answer_start is not None and answer_end is not None):
                        # Check if either start or end position matches
                        if (pred_start == answer_start or pred_start == answer_end or
                            pred_end == answer_start or pred_end == answer_end):
                            return 0.75
                        else:
                            return 0.5  # Chromosome match only
                    else:
                        return 0.5  # Chromosome match only (no positions available)
                else:
                    return 0.0

        elif task in ['SNP location', 'Gene location']:
            # Apply relaxed chromosome matching (matching evaluate.py)
            if isinstance(prediction, str) and isinstance(ground_truth, str):
                pred_chr = extract_chromosome(prediction)
                answer_chr = extract_chromosome(ground_truth)
                
                if prediction == ground_truth:
                    return 1.0
                elif pred_chr == answer_chr and pred_chr != "":
                    return 1.0  # Full score for chromosome match in these tasks
                else:
                    return 0.0

        else:
            # Exact match for other tasks
            return 1.0 if prediction == ground_truth else 0.0
            
    except Exception as e:
        print(f"Error calculating score for task {task}: {e}")
        return 0.0
    
    return 0.0

def extract_api_urls(prompting_logs: List) -> List[str]:
    """
    Extract API URLs from prompting logs
    """
    urls = []
    if not prompting_logs:
        return urls
    
    try:
        for log_entry in prompting_logs:
            if isinstance(log_entry, list) and len(log_entry) >= 2:
                # Look for URLs in the prompt/response text
                prompt_text = str(log_entry[0]) if log_entry[0] else ""
                response_text = str(log_entry[1]) if log_entry[1] else ""
                
                # Extract URLs using regex - match NCBI URLs
                url_pattern = r'https://[^\s\]\[\)]+\.(?:fcgi|cgi)[^\s\]\[\)]*'
                urls.extend(re.findall(url_pattern, prompt_text))
                urls.extend(re.findall(url_pattern, response_text))
    except Exception as e:
        print(f"Error extracting URLs: {e}")
    
    return list(set(urls))  # Remove duplicates

def load_ground_truth() -> Dict[str, Dict[str, Any]]:
    """
    Load ground truth data
    """
    try:
        qas = json.load(open('data/geneturing.json'))
        # Add GeneHop data if available
        try:
            genehop_data = json.load(open('data/genehop.json'))
            if 'Disease gene location' in genehop_data:
                qas['Disease gene location'] = genehop_data['Disease gene location']
            # Add other GeneHop tasks
            for task_name in ['SNP gene function', 'sequence gene alias']:
                if task_name in genehop_data:
                    qas[task_name] = genehop_data[task_name]
        except FileNotFoundError:
            print("Warning: genehop.json not found, skipping GeneHop tasks")
        return qas
    except FileNotFoundError:
        print("Error: geneturing.json not found")
        return {}

def load_results_from_folder(folder_path: str) -> Dict[str, List[Dict]]:
    """
    Load all result files from a folder
    """
    results = {}
    
    for task_file in glob.glob(os.path.join(folder_path, '*.json')):
        task_name = os.path.basename(task_file).replace('.json', '')
        
        try:
            with open(task_file, 'r') as f:
                preds = json.load(f)
            
            task_results = []
            for entry in preds:
                if len(entry) >= 3:
                    # Extract token usage information from logs (4th element)
                    if len(entry) > 3 and isinstance(entry[3], list):
                        total_tokens, total_cost, peak_context_usage, total_calls = extract_token_usage_info(entry[3])
                        stages_completed = extract_stages_completed(entry[3])
                    else:
                        total_tokens, total_cost, peak_context_usage, total_calls = "", "", "", ""
                        stages_completed = ""
                    
                    result = {
                        'question': entry[0],
                        'ground_truth': entry[1],
                        'prediction': entry[2],
                        'prompting_logs': entry[3] if len(entry) > 3 else [],
                        'elapsed_time': entry[5] if len(entry) > 5 else "",
                        'api_urls': extract_api_urls(entry[3] if len(entry) > 3 else []),
                        'total_tokens': total_tokens,
                        'total_cost': total_cost,
                        'peak_context_usage': peak_context_usage,
                        'total_calls': total_calls,
                        'stages_completed': stages_completed
                    }
                    task_results.append(result)
            
            results[task_name] = task_results
            
        except Exception as e:
            print(f"Error loading {task_file}: {e}")
            continue
    
    return results

def safe_numeric_conversion(value: Any, default_value: Any = "") -> Any:
    """
    Safely convert value to appropriate numeric type, return default if conversion fails
    """
    if value == "" or value is None:
        return default_value
    
    try:
        if isinstance(value, (int, float)):
            return value
        elif isinstance(value, str):
            if value.strip() == "":
                return default_value
            # Try int first, then float
            if '.' in value:
                return float(value)
            else:
                return int(value)
        else:
            return default_value
    except (ValueError, TypeError):
        return default_value

def compare_multiple_folders(folder_paths: List[str], details_csv: str, summary_csv: str):
    """
    Compare results from multiple folders with the first folder as baseline
    """
    if len(folder_paths) < 2:
        print("Error: Need at least 2 folders to compare")
        return
    
    print(f"Comparing results from {len(folder_paths)} folders:")
    for i, folder in enumerate(folder_paths):
        print(f"  Folder {i+1}: {folder}")
    print(f"Details will be saved to {details_csv}")
    print(f"Summary will be saved to {summary_csv}")
    
    # Load ground truth
    ground_truth_data = load_ground_truth()
    if not ground_truth_data:
        print("No ground truth data found. Exiting.")
        return
    
    # Load results from all folders
    all_results = []
    folder_names = []
    
    for folder_path in folder_paths:
        results = load_results_from_folder(folder_path)
        all_results.append(results)
        # Include parent directory to distinguish folders with same basename
        path_parts = os.path.normpath(folder_path).split(os.sep)
        if len(path_parts) >= 2:
            folder_name = f"{path_parts[-2]}~{path_parts[-1]}"  # parent~basename
        else:
            folder_name = os.path.basename(folder_path.rstrip('/\\'))
        folder_names.append(folder_name)
    
    # Prepare CSV data
    csv_data = []
    task_stats = {}
    
    # Get all tasks that exist in any folder
    all_tasks = set()
    for results in all_results:
        all_tasks.update(results.keys())
    
    for task in sorted(all_tasks):
        print(f"\nProcessing task: {task}")
        
        # Initialize task stats for all folders
        task_stats[task] = {
            'count': 0,
        }
        for i, folder_name in enumerate(folder_names):
            task_stats[task][f'score_{i}_sum'] = 0.0
            task_stats[task][f'score_{i}_count'] = 0
            task_stats[task][f'time_{i}_sum'] = 0.0
            task_stats[task][f'time_{i}_count'] = 0
            # Add token usage stats
            task_stats[task][f'tokens_{i}_sum'] = 0
            task_stats[task][f'tokens_{i}_count'] = 0
            task_stats[task][f'cost_{i}_sum'] = 0.0
            task_stats[task][f'cost_{i}_count'] = 0
            task_stats[task][f'peak_context_{i}_max'] = 0.0
            task_stats[task][f'calls_{i}_sum'] = 0
            task_stats[task][f'calls_{i}_count'] = 0
            task_stats[task][f'stages_{i}_sum'] = 0
            task_stats[task][f'stages_{i}_count'] = 0
        
        # Get ground truth for this task
        task_ground_truth = ground_truth_data.get(task, {})
        
        # Get results from all folders for this task
        task_results_all = []
        for results in all_results:
            task_results = {r['question']: r for r in results.get(task, [])}
            task_results_all.append(task_results)
        
        # Get all questions from all folders and ground truth
        all_questions = set(task_ground_truth.keys())
        for task_results in task_results_all:
            all_questions.update(task_results.keys())
        
        for question in sorted(all_questions):
            ground_truth = task_ground_truth.get(question, "")
            
            # Prepare CSV row
            csv_row = {
                'task': task,
                'question': question,
                'ground_truth': str(ground_truth),
            }
            
            # Process results from each folder
            predictions = []
            scores = []
            elapsed_times = []
            token_data = []
            
            for i, (task_results, folder_name) in enumerate(zip(task_results_all, folder_names)):
                result = task_results.get(question, {})
                
                # Get prediction and process it using imported get_answer function
                prediction = get_answer(result.get('prediction', ''), task) if result else ""
                elapsed_time = safe_float_conversion(result.get('elapsed_time', ''))
                api_urls = "; ".join(result.get('api_urls', [])) if result else ""
                
                # Extract token usage data
                total_tokens = safe_numeric_conversion(result.get('total_tokens', ''))
                total_cost = safe_numeric_conversion(result.get('total_cost', ''))
                peak_context_usage = safe_numeric_conversion(result.get('peak_context_usage', ''))
                total_calls = safe_numeric_conversion(result.get('total_calls', ''))
                stages_completed = safe_numeric_conversion(result.get('stages_completed', ''))
                
                # Calculate score
                score = calculate_score(prediction, ground_truth, task) if prediction else 0.0
                
                # Store for CSV and calculations
                predictions.append(prediction)
                scores.append(score)
                elapsed_times.append(elapsed_time)
                token_data.append({
                    'total_tokens': total_tokens,
                    'total_cost': total_cost,
                    'peak_context_usage': peak_context_usage,
                    'total_calls': total_calls,
                    'stages_completed': stages_completed
                })
                
                # Add to CSV row
                csv_row[f'prediction_{folder_name}'] = str(prediction)
                csv_row[f'score_{folder_name}'] = score
                csv_row[f'api_urls_{folder_name}'] = api_urls
                csv_row[f'elapsed_time_{folder_name}'] = elapsed_time
                csv_row[f'total_tokens_{folder_name}'] = total_tokens
                csv_row[f'total_cost_{folder_name}'] = total_cost
                csv_row[f'peak_context_usage_{folder_name}'] = peak_context_usage
                csv_row[f'total_calls_{folder_name}'] = total_calls
                csv_row[f'stages_completed_{folder_name}'] = stages_completed
                
                # Calculate differences relative to first folder (baseline)
                if i > 0:  # Skip first folder (baseline)
                    score_diff = scores[i] - scores[0]  # Current - baseline
                    time_diff = elapsed_times[i] - elapsed_times[0]  # Current - baseline
                    csv_row[f'score_diff_{folder_name}_vs_{folder_names[0]}'] = score_diff
                    csv_row[f'time_diff_{folder_name}_vs_{folder_names[0]}'] = time_diff
                    
                    # Token usage diffs - only if both sides have data
                    baseline_tokens = token_data[0]
                    current_tokens = token_data[i]
                    
                    if baseline_tokens['total_tokens'] != "" and current_tokens['total_tokens'] != "":
                        csv_row[f'tokens_diff_{folder_name}_vs_{folder_names[0]}'] = current_tokens['total_tokens'] - baseline_tokens['total_tokens']
                    else:
                        csv_row[f'tokens_diff_{folder_name}_vs_{folder_names[0]}'] = "N/A"
                        
                    if baseline_tokens['total_cost'] != "" and current_tokens['total_cost'] != "":
                        csv_row[f'cost_diff_{folder_name}_vs_{folder_names[0]}'] = current_tokens['total_cost'] - baseline_tokens['total_cost']
                    else:
                        csv_row[f'cost_diff_{folder_name}_vs_{folder_names[0]}'] = "N/A"
                        
                    if baseline_tokens['peak_context_usage'] != "" and current_tokens['peak_context_usage'] != "":
                        csv_row[f'peak_context_diff_{folder_name}_vs_{folder_names[0]}'] = current_tokens['peak_context_usage'] - baseline_tokens['peak_context_usage']
                    else:
                        csv_row[f'peak_context_diff_{folder_name}_vs_{folder_names[0]}'] = "N/A"
                        
                    if baseline_tokens['total_calls'] != "" and current_tokens['total_calls'] != "":
                        csv_row[f'calls_diff_{folder_name}_vs_{folder_names[0]}'] = current_tokens['total_calls'] - baseline_tokens['total_calls']
                    else:
                        csv_row[f'calls_diff_{folder_name}_vs_{folder_names[0]}'] = "N/A"
                        
                    if baseline_tokens['stages_completed'] != "" and current_tokens['stages_completed'] != "":
                        csv_row[f'stages_diff_{folder_name}_vs_{folder_names[0]}'] = current_tokens['stages_completed'] - baseline_tokens['stages_completed']
                    else:
                        csv_row[f'stages_diff_{folder_name}_vs_{folder_names[0]}'] = "N/A"
                
                # Update task statistics
                if prediction:
                    task_stats[task][f'score_{i}_sum'] += score
                    task_stats[task][f'score_{i}_count'] += 1
                if elapsed_time > 0:
                    task_stats[task][f'time_{i}_sum'] += elapsed_time
                    task_stats[task][f'time_{i}_count'] += 1
                    
                # Update token usage statistics
                if total_tokens != "":
                    task_stats[task][f'tokens_{i}_sum'] += total_tokens
                    task_stats[task][f'tokens_{i}_count'] += 1
                if total_cost != "":
                    task_stats[task][f'cost_{i}_sum'] += total_cost
                    task_stats[task][f'cost_{i}_count'] += 1
                if peak_context_usage != "":
                    # Convert percentage to decimal for comparison
                    peak_value = peak_context_usage / 100.0 if peak_context_usage > 1 else peak_context_usage
                    task_stats[task][f'peak_context_{i}_max'] = max(task_stats[task][f'peak_context_{i}_max'], peak_value)
                if total_calls != "":
                    task_stats[task][f'calls_{i}_sum'] += total_calls
                    task_stats[task][f'calls_{i}_count'] += 1
                if stages_completed != "":
                    task_stats[task][f'stages_{i}_sum'] += stages_completed
                    task_stats[task][f'stages_{i}_count'] += 1
            
            task_stats[task]['count'] += 1
            csv_data.append(csv_row)
    
    # Write detailed CSV file
    if csv_data:
        # Build dynamic fieldnames
        details_fieldnames = ['task', 'question', 'ground_truth']
        
        # Add prediction, score, api_urls, elapsed_time, and token usage columns for each folder
        for folder_name in folder_names:
            details_fieldnames.extend([
                f'prediction_{folder_name}',
                f'score_{folder_name}',
                f'api_urls_{folder_name}',
                f'elapsed_time_{folder_name}',
                f'total_tokens_{folder_name}',
                f'total_cost_{folder_name}',
                f'peak_context_usage_{folder_name}',
                f'total_calls_{folder_name}',
                f'stages_completed_{folder_name}'
            ])
        
        # Add difference columns (relative to first folder)
        for folder_name in folder_names[1:]:  # Skip first folder (baseline)
            details_fieldnames.extend([
                f'score_diff_{folder_name}_vs_{folder_names[0]}',
                f'time_diff_{folder_name}_vs_{folder_names[0]}',
                f'tokens_diff_{folder_name}_vs_{folder_names[0]}',
                f'cost_diff_{folder_name}_vs_{folder_names[0]}',
                f'peak_context_diff_{folder_name}_vs_{folder_names[0]}',
                f'calls_diff_{folder_name}_vs_{folder_names[0]}',
                f'stages_diff_{folder_name}_vs_{folder_names[0]}'
            ])
        
        with open(details_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=details_fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)
        
        print(f"\nDetailed comparison completed! Results saved to {details_csv}")
        print(f"Total rows: {len(csv_data)}")
    
    # Prepare and write summary CSV
    summary_data = []
    for task, stats in task_stats.items():
        summary_row = {
            'task': task,
            'total_questions': stats['count'],
        }
        
        # Calculate averages for each folder
        avg_scores = []
        avg_times = []
        avg_tokens = []
        avg_costs = []
        max_peak_contexts = []
        avg_calls = []
        avg_stages = []
        
        for i, folder_name in enumerate(folder_names):
            score_count = stats[f'score_{i}_count']
            time_count = stats[f'time_{i}_count']
            tokens_count = stats[f'tokens_{i}_count']
            cost_count = stats[f'cost_{i}_count']
            calls_count = stats[f'calls_{i}_count']
            stages_count = stats[f'stages_{i}_count']
            
            avg_score = stats[f'score_{i}_sum'] / score_count if score_count > 0 else 0.0
            avg_time = stats[f'time_{i}_sum'] / time_count if time_count > 0 else 0.0
            avg_token = stats[f'tokens_{i}_sum'] / tokens_count if tokens_count > 0 else 0.0
            avg_cost = stats[f'cost_{i}_sum'] / cost_count if cost_count > 0 else 0.0
            max_peak_context = stats[f'peak_context_{i}_max'] if stats[f'peak_context_{i}_max'] > 0 else 0.0
            avg_call = stats[f'calls_{i}_sum'] / calls_count if calls_count > 0 else 0.0
            avg_stage = stats[f'stages_{i}_sum'] / stages_count if stages_count > 0 else 0.0
            
            avg_scores.append(avg_score)
            avg_times.append(avg_time)
            avg_tokens.append(avg_token)
            avg_costs.append(avg_cost)
            max_peak_contexts.append(max_peak_context)
            avg_calls.append(avg_call)
            avg_stages.append(avg_stage)
            
            summary_row[f'questions_answered_{folder_name}'] = score_count
            summary_row[f'avg_score_{folder_name}'] = round(avg_score, 4)
            summary_row[f'avg_time_{folder_name}'] = round(avg_time, 2)
            summary_row[f'avg_tokens_{folder_name}'] = round(avg_token, 0) if avg_token > 0 else ""
            summary_row[f'avg_cost_{folder_name}'] = round(avg_cost, 6) if avg_cost > 0 else ""
            summary_row[f'max_peak_context_{folder_name}'] = round(max_peak_context, 4) if max_peak_context > 0 else ""
            summary_row[f'avg_calls_{folder_name}'] = round(avg_call, 1) if avg_call > 0 else ""
            summary_row[f'avg_stages_{folder_name}'] = round(avg_stage, 1) if avg_stage > 0 else ""
        
        # Calculate differences relative to first folder (baseline)
        for i, folder_name in enumerate(folder_names[1:], 1):  # Skip first folder
            score_diff = avg_scores[i] - avg_scores[0]
            time_diff = avg_times[i] - avg_times[0]
            summary_row[f'score_diff_{folder_name}_vs_{folder_names[0]}'] = round(score_diff, 4)
            summary_row[f'time_diff_{folder_name}_vs_{folder_names[0]}'] = round(time_diff, 2)
            
            # Token usage differences - only if both sides have data
            if avg_tokens[i] > 0 and avg_tokens[0] > 0:
                tokens_diff = avg_tokens[i] - avg_tokens[0]
                summary_row[f'tokens_diff_{folder_name}_vs_{folder_names[0]}'] = round(tokens_diff, 0)
            else:
                summary_row[f'tokens_diff_{folder_name}_vs_{folder_names[0]}'] = "N/A"
                
            if avg_costs[i] > 0 and avg_costs[0] > 0:
                cost_diff = avg_costs[i] - avg_costs[0]
                summary_row[f'cost_diff_{folder_name}_vs_{folder_names[0]}'] = round(cost_diff, 6)
            else:
                summary_row[f'cost_diff_{folder_name}_vs_{folder_names[0]}'] = "N/A"
                
            if max_peak_contexts[i] > 0 and max_peak_contexts[0] > 0:
                peak_context_diff = max_peak_contexts[i] - max_peak_contexts[0]
                summary_row[f'peak_context_diff_{folder_name}_vs_{folder_names[0]}'] = round(peak_context_diff, 4)
            else:
                summary_row[f'peak_context_diff_{folder_name}_vs_{folder_names[0]}'] = "N/A"
                
            if avg_calls[i] > 0 and avg_calls[0] > 0:
                calls_diff = avg_calls[i] - avg_calls[0]
                summary_row[f'calls_diff_{folder_name}_vs_{folder_names[0]}'] = round(calls_diff, 1)
            else:
                summary_row[f'calls_diff_{folder_name}_vs_{folder_names[0]}'] = "N/A"
                
            if avg_stages[i] > 0 and avg_stages[0] > 0:
                stages_diff = avg_stages[i] - avg_stages[0]
                summary_row[f'stages_diff_{folder_name}_vs_{folder_names[0]}'] = round(stages_diff, 1)
            else:
                summary_row[f'stages_diff_{folder_name}_vs_{folder_names[0]}'] = "N/A"
        
        summary_data.append(summary_row)
    
    # Write summary CSV
    if summary_data:
        # Build dynamic summary fieldnames
        summary_fieldnames = ['task', 'total_questions']
        
        # Add columns for each folder
        for folder_name in folder_names:
            summary_fieldnames.extend([
                f'questions_answered_{folder_name}',
                f'avg_score_{folder_name}',
                f'avg_time_{folder_name}',
                f'avg_tokens_{folder_name}',
                f'avg_cost_{folder_name}',
                f'max_peak_context_{folder_name}',
                f'avg_calls_{folder_name}',
                f'avg_stages_{folder_name}'
            ])
        
        # Add difference columns (relative to first folder)
        for folder_name in folder_names[1:]:  # Skip first folder (baseline)
            summary_fieldnames.extend([
                f'score_diff_{folder_name}_vs_{folder_names[0]}',
                f'time_diff_{folder_name}_vs_{folder_names[0]}',
                f'tokens_diff_{folder_name}_vs_{folder_names[0]}',
                f'cost_diff_{folder_name}_vs_{folder_names[0]}',
                f'peak_context_diff_{folder_name}_vs_{folder_names[0]}',
                f'calls_diff_{folder_name}_vs_{folder_names[0]}',
                f'stages_diff_{folder_name}_vs_{folder_names[0]}'
            ])
        
        with open(summary_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=summary_fieldnames)
            writer.writeheader()
            writer.writerows(summary_data)
        
        print(f"Summary comparison completed! Results saved to {summary_csv}")
        
        # Print summary statistics to console
        print(f"\nSummary Statistics (all comparisons relative to {folder_names[0]}):")
        print(f"Folders compared: {len(folder_names)}")
        print(f"\nPer-task averages:")
        for row in summary_data:
            task = row['task']
            print(f"{task}:")
            for folder_name in folder_names:
                score = row[f'avg_score_{folder_name}']
                time_val = row[f'avg_time_{folder_name}']
                count = row[f'questions_answered_{folder_name}']
                print(f"  {folder_name}: {score:.3f} score, {time_val:.2f}s avg time ({count} questions)")
            
            # Show differences relative to baseline
            for folder_name in folder_names[1:]:
                score_diff = row[f'score_diff_{folder_name}_vs_{folder_names[0]}']
                if score_diff != 0:
                    direction = "better" if score_diff > 0 else "worse"
                    print(f"  → {folder_name} performs {direction} than {folder_names[0]} by {abs(score_diff):.3f}")
    else:
        print("No data to write to summary CSV")

def generate_default_output_paths(folder_paths: List[str]) -> Tuple[str, str]:
    """
    Generate default output CSV paths based on folder names using $ separator
    Places evaluation_reports in the parent directory of the first folder
    Returns: (details_csv_path, summary_csv_path)
    """
    # Extract folder names (remove trailing slashes and get basename)
    folder_names = []
    for folder_path in folder_paths:
        path_parts = os.path.normpath(folder_path).split(os.sep)
        if len(path_parts) >= 2:
            folder_name = f"{path_parts[-2]}~{path_parts[-1]}"
        else:
            folder_name = os.path.basename(folder_path.rstrip('/\\'))
        folder_names.append(folder_name)
        
    # Get the parent directory of the first folder
    parent_dir = os.path.dirname(os.path.abspath(folder_paths[0]))
    
    # Create evaluation_reports directory in the parent of the first folder
    reports_dir = os.path.join(parent_dir, "evaluation_reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    # Generate filenames using $ separator
    folder_suffix = "$".join(folder_names)
    details_filename = f"details${folder_suffix}.csv"
    summary_filename = f"summary${folder_suffix}.csv"
    
    details_path = os.path.join(reports_dir, details_filename)
    summary_path = os.path.join(reports_dir, summary_filename)
    
    return details_path, summary_path

def safe_float_conversion(value: Any) -> float:
    """
    Safely convert value to float, return 0.0 if conversion fails
    """
    try:
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, str):
            if value.strip() == "":
                return 0.0
            return float(value)
        else:
            return 0.0
    except (ValueError, TypeError):
        return 0.0
    
if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python compare_results.py <folder_A> <folder_B> [folder_C] [folder_D] [details_output.csv] [summary_output.csv]")
        print("Examples:")
        print("  python compare_results.py results/A results/B")
        print("  python compare_results.py results/A results/B results/C")
        print("  python compare_results.py results/A results/B results/C results/D")
        print("  python compare_results.py results/A results/B custom_details.csv custom_summary.csv")
        print("\nAll comparisons are relative to the first folder (baseline)")
        print("If output files are not provided, they will be saved as:")
        print("  evaluation_reports/details$A$B$C$D.csv")
        print("  evaluation_reports/summary$A$B$C$D.csv")
        sys.exit(1)
    
    # Parse arguments to separate folder paths from output file paths
    args = sys.argv[1:]
    
    # Find the split point: last 2 args that end with .csv are output files
    folder_paths = []
    output_files = []
    
    # Check if last 1 or 2 arguments are CSV files
    if len(args) >= 2 and args[-1].endswith('.csv') and args[-2].endswith('.csv'):
        # Both details and summary files provided
        folder_paths = args[:-2]
        output_files = args[-2:]
    elif len(args) >= 1 and args[-1].endswith('.csv'):
        print("Error: If providing custom output files, you must specify both details and summary files")
        sys.exit(1)
    else:
        # No output files provided, use all args as folder paths
        folder_paths = args
        output_files = []
    
    # Validate minimum number of folders
    if len(folder_paths) < 2:
        print("Error: Need at least 2 folders to compare")
        sys.exit(1)
    
    # Validate maximum number of folders
    if len(folder_paths) > 4:
        print("Error: Maximum 4 folders supported")
        sys.exit(1)
    
    # Check if all folders exist
    for folder_path in folder_paths:
        if not os.path.isdir(folder_path):
            print(f"Error: Folder {folder_path} does not exist")
            sys.exit(1)
    
    # Determine output file paths
    if output_files:
        details_csv, summary_csv = output_files
    else:
        details_csv, summary_csv = generate_default_output_paths(folder_paths)
        print(f"No output files specified. Using defaults:")
        print(f"  Details: {details_csv}")
        print(f"  Summary: {summary_csv}")
    
    compare_multiple_folders(folder_paths, details_csv, summary_csv)