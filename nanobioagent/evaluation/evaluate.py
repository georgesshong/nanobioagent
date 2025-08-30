'''
evaluate GeneGPT on all GeneTuring tasks and one GeneHop task (Disease gene location)
with additional detailed reporting, relaxed chromosome matching, and gene symbol normalization
'''

import glob
import json
import os
import sys
import csv
import re
from collections import defaultdict

def extract_chromosome_positions(chromosome_str):
    """
    Extract chromosome and start/end positions from a chromosome location string.
    Examples:
    - "chr1:171187-171287" -> ("chr1", 171187, 171287)
    - "chr7" -> ("chr7", None, None)
    - "Chromosome 14 NC_000014.9 (60981245..61083733)" -> ("chr14", 60981245, 61083733)
    Returns: (chromosome, start_pos, end_pos)
    """
    if not chromosome_str:
        return ("", None, None)
    
    chromosome_str = str(chromosome_str).strip()
    
    # Extract basic chromosome part
    base_chr = extract_chromosome(chromosome_str)
    
    # Look for colon-dash format: chr1:171187-171287
    colon_dash_match = re.search(r'chr[0-9XYxy]+:(\d+)-(\d+)', chromosome_str, re.IGNORECASE)
    if colon_dash_match:
        start_pos = int(colon_dash_match.group(1))
        end_pos = int(colon_dash_match.group(2))
        return (base_chr, start_pos, end_pos)
    
    # Look for parentheses format: (60981245..61083733)
    paren_match = re.search(r'\((\d+)\.\.(\d+)\)', chromosome_str)
    if paren_match:
        start_pos = int(paren_match.group(1))
        end_pos = int(paren_match.group(2))
        return (base_chr, start_pos, end_pos)
    
    # No positions found
    return (base_chr, None, None)

def extract_chromosome(chromosome_str):
    """
    Extract just the basic chromosome part (e.g., chrX, chr7) from detailed chromosome location.
    This makes the matching more relaxed to handle cases like chr7 vs chr7p14.1
    """
    if not chromosome_str:
        return ""
    
    # Handle different input formats
    chromosome_str = str(chromosome_str).strip()
    
    # If it's just a chromosome like "chr8" or "Chromosome 8"
    if chromosome_str.lower().startswith('chromosome'):
        # Extract number/letter after "chromosome"
        match = re.search(r'chromosome\s+([0-9XYxy]+)', chromosome_str, re.IGNORECASE)
        if match:
            return f"chr{match.group(1)}"
    
    # If it already starts with chr (like "chr15:91950805-91950932")
    if chromosome_str.lower().startswith('chr'):
        # Extract just the chr part before any colon or additional info
        match = re.match(r'(chr[0-9XYxy]+)', chromosome_str, re.IGNORECASE)
        if match:
            return match.group(1).lower()
    
    # For complex cases like "Chromosome Xq21.2-22.2"
    if 'x' in chromosome_str.lower() and ('q' in chromosome_str.lower() or 'p' in chromosome_str.lower()):
        return "chrx"
    
    # Try to extract any chromosome pattern
    match = re.search(r'([0-9XYxy]+)', chromosome_str)
    if match:
        return f"chr{match.group(1).lower()}"
    
    return chromosome_str.lower()

def split_comma_separated(text):
    if ', ' in text:
        return text.split(', ')    # Split on comma+space
    else:
        return [item.strip() for item in text.split(',')]  # Split on comma only

def normalize_gene_symbol(gene_symbol):
    """
    Normalize gene symbols by removing descriptive parts in parentheses.
    Examples:
    - "GNMT (Glycine N-methyltransferase)" -> "GNMT"
    - "SGOL1 (Shugoshin-Like 1)" -> "SGOL1"
    - "MNX1 (Motor Neuron and Pancreas Homeobox 1)" -> "MNX1"
    
    This function also handles cases where the gene symbol itself might be in parentheses.
    """
    if not gene_symbol:
        return gene_symbol
        
    # Remove any descriptive parts in parentheses, pattern: SYMBOL (Description)
    clean_symbol = re.sub(r'\s*\([^)]*\)', '', gene_symbol).strip()
    return clean_symbol

def extract_token_usage_info(logs):
    """
    Extract token usage information from the logs array.
    Returns: (total_tokens, total_cost, peak_context_usage, total_calls)
    """
    if not logs or not isinstance(logs, list):
        return "", "", "", ""
    
    total_tokens = ""
    total_cost = ""
    peak_context_usage = ""
    total_calls = ""
    
    # Look for the token usage section
    in_token_section = False
    for log in logs:
        if not isinstance(log, str):
            continue
            
        if "=== Token Usage ===" in log:
            in_token_section = True
            continue
        elif log.startswith("===") and in_token_section:
            # End of token usage section
            break
        elif in_token_section:
            # Parse the token usage lines
            if log.startswith("Total Calls:"):
                match = re.search(r'Total Calls:\s*(\d+)', log)
                if match:
                    total_calls = int(match.group(1))
            elif log.startswith("Total Tokens:"):
                match = re.search(r'Total Tokens:\s*([\d,]+)', log)
                if match:
                    # Remove commas and convert to int
                    total_tokens = int(match.group(1).replace(',', ''))
            elif log.startswith("Total Cost:"):
                match = re.search(r'Total Cost:\s*\$?([\d.]+)', log)
                if match:
                    total_cost = float(match.group(1))
            elif log.startswith("Peak Context Usage:"):
                match = re.search(r'Peak Context Usage:\s*([\d.]+)%', log)
                if match:
                    peak_context_usage = float(match.group(1))
    
    return total_tokens, total_cost, peak_context_usage, total_calls

def extract_stages_completed(logs):
    """
    Extract the maximum stage completed from the logs array.
    Returns: integer representing max stage reached (1, 2, 3, or 4)
    """
    if not logs or not isinstance(logs, list):
        return ""
    
    max_stage = 0
    
    for log in logs:
        if not isinstance(log, str):
            continue
            
        # Look for stage markers
        if "Stage 1:" in log:
            max_stage = max(max_stage, 1)
        elif "Stage 2:" in log:
            max_stage = max(max_stage, 2)
        elif "Stage 3:" in log:
            max_stage = max(max_stage, 3)
        elif "Stage 4:" in log:
            max_stage = max(max_stage, 4)
    
    return max_stage if max_stage > 0 else ""

def is_excluded_answer(answer):
    """Check if an answer should be excluded from scoring (marked with '?' prefix)"""
    if isinstance(answer, str):
        return answer.startswith('?')
    elif isinstance(answer, list):
        # For list answers, check if any element starts with '?'
        return any(str(item).startswith('?') for item in answer)
    return False

def clean_excluded_answer(answer):
    """Remove the '?' prefix from excluded answers for display purposes"""
    if isinstance(answer, str) and answer.startswith('?'):
        return answer[1:]  # Remove the '?' prefix
    elif isinstance(answer, list):
        return [str(item)[1:] if str(item).startswith('?') else str(item) for item in answer]
    return answer

def get_answer(answer, task):

    mapper = {
                'Caenorhabditis elegans': 'worm',
                'Homo sapiens': 'human',
                'Homo sapiens (human)': 'human',
                'Zebrafish (Danio rerio)': 'zebrafish',
                'Danio rerio': 'zebrafish',
                'Zebrafish': 'zebrafish',
                'Mus musculus': 'mouse',
                'Mus musculus (house mouse)': 'mouse',
                'Saccharomyces cerevisiae': 'yeast',
                "Saccharomyces cerevisiae (baker's yeast)": 'yeast',
                'Saccharomyces cerevisiae (yeast)': 'yeast',
                'Rattus norvegicus': 'rat',
                'Rattus norvegicus (rat)': 'rat',
                'Gallus gallus': 'chicken',
                'Gallus gallus (chicken)': 'chicken',
                'Meleagris gallopavo': 'turkey',
                'Cairina moschata': 'duck',
                'Cairina moschata breed yongchun': 'duck'
            }

    if task == 'SNP location':
        answer = answer.strip().replace('Answer: ', '')
        # Check for the pattern "chromosome X" in the text
        chromosome_match = re.search(r'chromosome\s+([0-9XYxy]+)', answer)
        if chromosome_match:
            # Extract the chromosome number/letter and format it as chrX
            answer = 'chr' + chromosome_match.group(1)
        else:
            # Fall back to original behavior
            answer_parts = answer.split()
            answer = answer_parts[-1] if answer_parts else ""
            if 'chr' not in answer:
                answer = 'chr' + answer

    elif task == 'Gene location':
        answer = answer.strip().replace('Answer: ', '')
        # Check for the pattern "chromosome X" in the text
        chromosome_match = re.search(r'chromosome\s+([0-9XYxy]+)', answer)
        if chromosome_match:
            # Extract the chromosome number/letter and format it as chrX
            answer = 'chr' + chromosome_match.group(1)
        else:
            # Fall back to original behavior with safety check
            answer_parts = answer.split()
            if answer_parts:  # Make sure we have at least one word
                answer = answer_parts[-1]
                if 'chr' not in answer:
                    answer = 'chr' + answer
            else:
                # Handle case where answer is empty or only whitespace
                answer = 'chrUNKNOWN'

    elif task == 'Gene disease association':
        answer = answer.strip().replace('Answer: ', '')
        answer = split_comma_separated(answer)
        
    elif task == 'Disease gene location':
        answer = answer.strip().replace('Answer: ', '')
        answer = split_comma_separated(answer)
        # Normalize each gene symbol if present in chromosome location
        # Note: This task usually returns chromosome locations, not gene symbols

    elif task == 'Protein-coding genes':
        answer = answer.strip().replace('Answer: ', '')
        if answer.startswith('Yes'):
            answer = 'TRUE'
        elif answer.startswith('No'):
            answer = 'NA'

    elif task == 'Multi-species DNA aligment':
        answer = answer.strip().replace('Answer: ', '')
        # Normalize each animal
        answer = normalize_gene_symbol(answer)

        answer = mapper.get(answer, answer)

    else:
        answer = answer.strip().replace('Answer: ', '')
    
    return answer

def run_evaluation(folder, qas_file, output_suffix="", use_updated_logic=False):
    """Run evaluation with specified parameters"""
    print(f"\n{'='*50}")
    print(f"Running evaluation with {qas_file}")
    print(f"Output suffix: {output_suffix}")
    print(f"Updated logic: {use_updated_logic}")
    print(f"{'='*50}")
    
    # Load the QA data
    qas = json.load(open(qas_file))
    
    # Add genehop data if available
    genehop_file = 'data/genehop.json'
    if os.path.exists(genehop_file):
        genehop_data = json.load(open(genehop_file))
        if 'Disease gene location' in genehop_data:
            qas['Disease gene location'] = genehop_data['Disease gene location']
    
    # Create detailed report structures
    all_results = []
    task_summaries = defaultdict(lambda: {'total': 0, 'correct': 0, 'score': 0.0})
    
    for task_file in glob.glob(os.path.join(folder, '*.json')):
        task = os.path.basename(task_file).replace('.json', '')
        print(f'\nEvaluating {task}')
        
        preds = json.load(open(task_file))
        
        if task not in qas:
            print(f'{task} is not automatically evaluated.')
            continue
        
        info = qas[task]
        pred_q2a = {}
        
        for entry in preds:
            pred_q2a[entry[0]] = get_answer(entry[2], task)
        
        correct = []
        question_results = []
        excluded_count = 0  # Track excluded questions
        
        for question, answer in info.items():
            result = {'task': task, 'question': question, 'ground_truth': answer}
            
            # Check if this question should be excluded from scoring
            should_exclude = is_excluded_answer(answer)
            if should_exclude:
                excluded_count += 1
                # Clean the answer for display (remove '?' prefix)
                result['ground_truth'] = clean_excluded_answer(answer)
                
            if question in pred_q2a:
                prediction = pred_q2a[question]
                result['prediction'] = prediction
                
                # Find the entry for this question to extract additional data
                for entry in preds:
                    if entry[0] == question:
                        # Add API URLs (5th element) if it exists
                        if len(entry) > 4 and isinstance(entry[4], list):
                            result['api_urls'] = ",".join(entry[4])
                        else:
                            result['api_urls'] = ""
                            
                        # Add number of model calls (length of 4th element)
                        if len(entry) > 3 and isinstance(entry[3], list):
                            result['num_model_calls'] = len(entry[3])
                        else:
                            result['num_model_calls'] = ""
                            
                        # Add elapsed time (6th element) if it exists
                        if len(entry) > 5:
                            result['elapsed_time'] = entry[5]
                        else:
                            result['elapsed_time'] = ""
                        
                        # Extract token usage information from logs (4th element)
                        if len(entry) > 3 and isinstance(entry[3], list):
                            total_tokens, total_cost, peak_context_usage, total_calls = extract_token_usage_info(entry[3])
                            result['total_tokens'] = total_tokens
                            result['total_cost'] = total_cost
                            result['peak_context_usage'] = peak_context_usage
                            result['total_calls'] = total_calls
                            
                            # Extract stages completed information
                            result['stages_completed'] = extract_stages_completed(entry[3])
                        else:
                            result['total_tokens'] = ""
                            result['total_cost'] = ""
                            result['peak_context_usage'] = ""
                            result['total_calls'] = ""
                            result['stages_completed'] = ""
                            
                        break

                # Only calculate score if not excluded
                if should_exclude:
                    result['score'] = 'EXCLUDED'
                    result['success'] = 'EXCLUDED'
                    # Don't add to correct array - this excludes it from score calculation
                else:
                    
                    if task == 'Gene disease association':
                        # Get the original ground truth answer before any normalization
                        orig_answer = answer
                        if isinstance(orig_answer, str):
                            orig_answer = split_comma_separated(orig_answer)
                        
                        # Normalize answers and prediction for comparison
                        normalized_answer = [normalize_gene_symbol(gene) for gene in orig_answer]
                        
                        # Handle both list and string predictions
                        if isinstance(prediction, list):
                            normalized_prediction = [normalize_gene_symbol(gene) for gene in prediction]
                        else:
                            # If prediction is a string, split and normalize
                            prediction_list = split_comma_separated(prediction) if isinstance(prediction, str) else []
                            normalized_prediction = [normalize_gene_symbol(gene) for gene in prediction_list]
                        
                        # Check for matches using normalized values
                        answer_in = [ans in normalized_prediction for ans in normalized_answer]
                        
                        # Store normalized values for debugging
                        result['normalized_answer'] = ', '.join(normalized_answer)
                        result['normalized_prediction'] = ', '.join(normalized_prediction) if normalized_prediction else ''
                        
                        score = sum(answer_in) / len(answer_in)
                        correct.append(score)
                        result['score'] = score
                        result['success'] = 'Partial' if 0 < score < 1 else ('Yes' if score == 1 else 'No')
                    
                    elif task == 'Disease gene location':
                        if isinstance(prediction, list):
                            answer_in = [ans in prediction for ans in answer]
                        else:
                            # Handle the case where prediction is not a list
                            prediction_list = split_comma_separated(prediction) if isinstance(prediction, str) else []
                            answer_in = [ans in prediction_list for ans in answer]
                        
                        score = sum(answer_in) / len(answer_in)
                        correct.append(score)
                        result['score'] = score
                        result['success'] = 'Partial' if 0 < score < 1 else ('Yes' if score == 1 else 'No')
                    
                    elif task == 'Human genome DNA aligment':
                        pred = prediction
                        pred_chr, pred_start, pred_end = extract_chromosome_positions(pred)
                        answer_chr, answer_start, answer_end = extract_chromosome_positions(answer)
                        
                        if pred == answer:
                            score = 1
                            result['success'] = 'Yes'
                        elif pred_chr == answer_chr and pred_chr != "":
                            # Check for position matches if both have positions
                            if (pred_start is not None and pred_end is not None and 
                                answer_start is not None and answer_end is not None):
                                # Check if either start or end position matches
                                if (pred_start == answer_start or pred_start == answer_end or
                                    pred_end == answer_start or pred_end == answer_end):
                                    score = 0.75
                                    result['success'] = 'Yes (position match)'
                                elif use_updated_logic:
                                    # Updated logic: chromosome match = full score
                                    score = 0.500001 # 1
                                    result['success'] = 'Yes (chromosome match)'
                                else:
                                    # Original logic: chromosome match = partial score
                                    score = 0.5
                                    result['success'] = 'Partial'
                            else:
                                # No positions available, fall back to chromosome matching
                                if use_updated_logic:
                                    # Updated logic: chromosome match = full score
                                    score = 0.500001 # 1
                                    result['success'] = 'Yes (chromosome match)'
                                else:
                                    # Original logic: chromosome match = partial score
                                    score = 0.5
                                    result['success'] = 'Partial'
                        else:
                            score = 0
                            result['success'] = 'No'
                        
                        correct.append(score)
                        result['score'] = score
                    
                    elif task == 'SNP location' or task == 'Gene location':
                        # Apply relaxed chromosome matching
                        pred_chr = extract_chromosome(prediction)
                        answer_chr = extract_chromosome(answer)
                        
                        if prediction == answer:
                            score = 1
                            result['success'] = 'Yes'
                        elif pred_chr == answer_chr:
                            # Consider it correct if the basic chromosome part matches
                            score = 1
                            result['success'] = 'Yes (relaxed)'
                        else:
                            score = 0
                            result['success'] = 'No'
                        
                        correct.append(score)
                        result['score'] = score
                        
                        # Add debug info for chromosome matching
                        result['extracted_pred_chr'] = pred_chr
                        result['extracted_answer_chr'] = answer_chr
                    
                    else:
                        if prediction == answer:
                            score = 1
                            result['success'] = 'Yes'
                        else:
                            score = 0
                            result['success'] = 'No'
                        
                        correct.append(score)
                        result['score'] = score
            else:
                # Question was not answered
                result['prediction'] = 'N/A'
                result['api_urls'] = ""
                result['num_model_calls'] = ""
                result['elapsed_time'] = ""
                result['total_tokens'] = ""
                result['total_cost'] = ""
                result['peak_context_usage'] = ""
                result['total_calls'] = ""
                result['stages_completed'] = ""
                if should_exclude:
                    result['score'] = 'EXCLUDED'
                    result['success'] = 'EXCLUDED'
                    # Don't add to correct array
                else:
                    result['score'] = 0
                    result['success'] = 'Missing'
                    correct.append(0)
            
            question_results.append(result)
        
        # Calculate task summary (excluding the excluded questions)
        total_score = sum(correct) / len(correct) if correct else 0
        effective_total = len(info) - excluded_count  # Total minus excluded
        
        print(f"Score: {total_score:.4f} ({len(correct)}/{effective_total} questions, {excluded_count} excluded)")
        
        task_summaries[task]['total'] = effective_total
        task_summaries[task]['correct'] = sum(correct)
        task_summaries[task]['score'] = total_score
        task_summaries[task]['excluded'] = excluded_count  # Track exclusions
        
        all_results.extend(question_results)
    
    # Output detailed report as CSV
    output_dir = os.path.join(os.path.dirname(folder), 'evaluation_reports')
    os.makedirs(output_dir, exist_ok=True)
    
    # Write individual results
    details_file = os.path.join(output_dir, f'{os.path.basename(folder)}_details{output_suffix}.csv')
    with open(details_file, 'w', newline='', encoding='utf-8', errors='replace') as f:
        fieldnames = ['task', 'question', 'ground_truth', 'prediction', 'score', 'success', 
                    'api_urls', 'num_model_calls', 'elapsed_time', 'total_tokens', 'total_cost', 
                    'peak_context_usage', 'total_calls', 'stages_completed']
        
        # Add chromosome extraction fields for chromosome-related tasks
        if any(result.get('extracted_pred_chr') for result in all_results):
            fieldnames.extend(['extracted_pred_chr', 'extracted_answer_chr'])
            
        # Add normalized gene fields for gene association tasks
        if any(result.get('normalized_answer') for result in all_results):
            fieldnames.extend(['normalized_answer', 'normalized_prediction'])
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            # Filter out any keys not in fieldnames
            filtered_result = {k: v for k, v in result.items() if k in fieldnames}
            writer.writerow(filtered_result)
    
    # Calculate task-level aggregates for the new columns
    task_aggregates = defaultdict(lambda: {
        'total_tokens_sum': 0, 'total_tokens_count': 0,
        'total_cost_sum': 0.0, 'total_cost_count': 0,
        'peak_context_usage_max': 0.0,
        'total_calls_sum': 0, 'total_calls_count': 0,
        'stages_completed_sum': 0, 'stages_completed_count': 0,
        'elapsed_time_sum': 0.0, 'elapsed_time_count': 0
    })
    
    for result in all_results:
        task = result['task']
        # Only include answered questions (not N/A)
        if result['prediction'] != 'N/A':
            if result['total_tokens'] != "":
                task_aggregates[task]['total_tokens_sum'] += result['total_tokens']
                task_aggregates[task]['total_tokens_count'] += 1
            if result['total_cost'] != "":
                task_aggregates[task]['total_cost_sum'] += result['total_cost']
                task_aggregates[task]['total_cost_count'] += 1
            if result['peak_context_usage'] != "":
                # Convert percentage to decimal (e.g., 1.56% -> 0.0156)
                peak_value = result['peak_context_usage'] / 100.0
                task_aggregates[task]['peak_context_usage_max'] = max(task_aggregates[task]['peak_context_usage_max'], peak_value)
            if result['total_calls'] != "":
                task_aggregates[task]['total_calls_sum'] += result['total_calls']
                task_aggregates[task]['total_calls_count'] += 1
            if result['stages_completed'] != "":
                task_aggregates[task]['stages_completed_sum'] += result['stages_completed']
                task_aggregates[task]['stages_completed_count'] += 1
            if result['elapsed_time'] != "":
                task_aggregates[task]['elapsed_time_sum'] += result['elapsed_time']
                task_aggregates[task]['elapsed_time_count'] += 1
    
    # Write task summary
    summary_file = os.path.join(output_dir, f'{os.path.basename(folder)}_summary{output_suffix}.csv')
    with open(summary_file, 'w', newline='', encoding='utf-8', errors='replace') as f:
        fieldnames = ['task', 'total_questions', 'correct_answers', 'score', 'avg_total_tokens', 
                     'avg_total_cost', 'max_peak_context_usage', 'avg_total_calls', 'avg_stages_completed', 'sum_elapsed_time']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for task, summary in task_summaries.items():
            agg = task_aggregates[task]
            
            # Calculate averages and aggregates
            avg_total_tokens = agg['total_tokens_sum'] / agg['total_tokens_count'] if agg['total_tokens_count'] > 0 else ""
            avg_total_cost = agg['total_cost_sum'] / agg['total_cost_count'] if agg['total_cost_count'] > 0 else ""
            max_peak_context_usage = agg['peak_context_usage_max'] if agg['peak_context_usage_max'] > 0 else ""
            avg_total_calls = agg['total_calls_sum'] / agg['total_calls_count'] if agg['total_calls_count'] > 0 else ""
            avg_stages_completed = agg['stages_completed_sum'] / agg['stages_completed_count'] if agg['stages_completed_count'] > 0 else ""
            sum_elapsed_time = agg['elapsed_time_sum'] if agg['elapsed_time_count'] > 0 else ""
            
            writer.writerow({
                'task': task,
                'total_questions': summary['total'],
                'correct_answers': summary['correct'],
                'score': f"{summary['score']:.4f}",
                'avg_total_tokens': f"{avg_total_tokens:.0f}" if avg_total_tokens != "" else "",
                'avg_total_cost': f"{avg_total_cost:.6f}" if avg_total_cost != "" else "",
                'max_peak_context_usage': f"{max_peak_context_usage:.6f}" if max_peak_context_usage != "" else "",
                'avg_total_calls': f"{avg_total_calls}" if avg_total_calls != "" else "",
                'avg_stages_completed': f"{avg_stages_completed}" if avg_stages_completed != "" else "",
                'sum_elapsed_time': f"{sum_elapsed_time:.2f}" if sum_elapsed_time != "" else ""
            })
    
    # Calculate overall score
    all_scores = [summary['score'] for summary in task_summaries.values()]
    overall_score = sum(all_scores) / len(all_scores) if all_scores else 0
    
    print(f"\n=== EVALUATION SUMMARY{output_suffix.upper()} ===")
    print(f"Overall Score: {overall_score:.4f}")
    print(f"Detailed reports saved to: {output_dir}")
    print(f"- Question-level details: {os.path.basename(details_file)}")
    print(f"- Task-level summary: {os.path.basename(summary_file)}")
    
    return task_summaries, overall_score

def main_evaluation(folder_path, qas_file='data/geneturing.json', use_updated_logic=True):
    """
    Main evaluation function that can be called programmatically
    Args:
        folder_path: Path to the results folder containing JSON files
        qas_file: Path to the geneturing.json file (default: 'data/geneturing.json')
        use_updated_logic: Whether to use updated logic by default (default: True)
    """
    # Check if required files exist
    original_qas = qas_file
    updated_qas = qas_file.replace('.json', '_updated.json')
    
    if not os.path.exists(original_qas):
        print(f"Error: {original_qas} not found!")
        return
    
    # Run original evaluation
    print("RUNNING ORIGINAL EVALUATION")
    original_results, original_overall = run_evaluation(
        folder=folder_path,
        qas_file=original_qas,
        output_suffix="",
        use_updated_logic=False
    )
    
    # Run updated evaluation if file exists
    if os.path.exists(updated_qas):
        print("\n\nRUNNING UPDATED EVALUATION")
        updated_results, updated_overall = run_evaluation(
            folder=folder_path,
            qas_file=updated_qas,
            output_suffix="_updated",
            use_updated_logic=use_updated_logic
        )
        
        # Compare results
        print(f'\n{"="*70}')
        print("COMPARISON SUMMARY")
        print(f'{"="*70}')
        print(f"{'Task':<35} {'Original':<12} {'Updated':<12} {'Difference':<12}")
        print(f'{"-"*70}')
        
        all_tasks = set(original_results.keys()) | set(updated_results.keys())
        for task in sorted(all_tasks):
            orig_score = original_results.get(task, {}).get('score', 0.0)
            upd_score = updated_results.get(task, {}).get('score', 0.0)
            diff = upd_score - orig_score
            print(f"{task:<35} {orig_score:<12.4f} {upd_score:<12.4f} {diff:+.4f}")
        
        print(f'{"-"*70}')
        print(f"{'OVERALL':<35} {original_overall:<12.4f} {updated_overall:<12.4f} {updated_overall-original_overall:+.4f}")
        print(f'{"="*70}')
        
        # Save comparison report
        output_dir = os.path.join(os.path.dirname(folder_path), 'evaluation_reports')
        comparison_file = os.path.join(output_dir, f'{os.path.basename(folder_path)}_comparison.csv')
        with open(comparison_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['task', 'original_score', 'updated_score', 'difference', 'improvement']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for task in sorted(all_tasks):
                orig_score = original_results.get(task, {}).get('score', 0.0)
                upd_score = updated_results.get(task, {}).get('score', 0.0)
                diff = upd_score - orig_score
                improvement = 'Yes' if diff > 0 else ('No' if diff < 0 else 'Same')
                
                writer.writerow({
                    'task': task,
                    'original_score': f"{orig_score:.4f}",
                    'updated_score': f"{upd_score:.4f}",
                    'difference': f"{diff:+.4f}",
                    'improvement': improvement
                })
            
            # Add overall row
            writer.writerow({
                'task': 'OVERALL',
                'original_score': f"{original_overall:.4f}",
                'updated_score': f"{updated_overall:.4f}",
                'difference': f"{updated_overall-original_overall:+.4f}",
                'improvement': 'Yes' if updated_overall > original_overall else ('No' if updated_overall < original_overall else 'Same')
            })
        
        print(f"Comparison report saved to: {os.path.basename(comparison_file)}")
        
    else:
        print(f"\nWarning: {updated_qas} not found. Only running original evaluation.")

if __name__ == '__main__':
    # Command line interface - result dir path to evaluate
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <folder_path>")
        sys.exit(1)
    
    folder = sys.argv[1]
    main_evaluation(folder)