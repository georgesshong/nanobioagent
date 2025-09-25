'''
evaluate GeneGPT on all GeneTuring tasks and one GeneHop task (Disease gene location)
with additional detailed reporting, relaxed chromosome matching, and gene symbol normalization
'''
DEBUG_PRINT = False
import glob
import json
import os
import sys
import csv
import re
from collections import defaultdict

# Define the task type mapping for classification scoring
TASK_TYPE_MAPPING = {
    'Gene alias': ['gene_alias'],
    'Gene disease association': ['gene_disease_association'], 
    'Gene location': ['gene_location'],
    'SNP location': ['gene_snp_association', 'gene_location'],  # Can be solved by either
    'Gene SNP association': ['gene_snp_association'],
    'Protein-coding genes': ['protein-coding_genes'],
    'Human genome DNA aligment': ['human_genome_dna_alignment'], # mis-spelled in original gene turing test in genegpt paper, kept for backward compatibility
    'Multi-species DNA aligment': ['multi-species_dna_alignment'],
    'Gene name conversion': ['gene_alias'],
    'ISM sequence optimization': ['ism_sequence_optimization'],
    'SNP splicing analysis': ['snp_splicing_analysis'], 
    'Tissue expression comparison': ['tissue_expression_comparison']
}

def extract_task_classification_from_logs(logs):
    """
    Extract task classification from logs array.
    Returns: (predicted_task_type, classification_found, error_occurred)
    """
    predicted_task_type = None
    classification_found = False
    error_occurred = False
    
    if not logs or not isinstance(logs, list):
        return predicted_task_type, classification_found, error_occurred
    
    for log_entry in logs:
        if isinstance(log_entry, str):
            # Look for "Task classified as: XXXX" pattern
            match = re.search(r'Task classified as:\s*([a-zA-Z0-9_-]+)', log_entry)
            if match:
                predicted_task_type = match.group(1)
                classification_found = True
                break
            
            # Check for errors in classification stage
            if 'error' in log_entry.lower() or 'exception' in log_entry.lower():
                if 'classification' in log_entry.lower() or 'Stage 1' in log_entry:
                    error_occurred = True
    
    return predicted_task_type, classification_found, error_occurred

def score_task_classification_for_question(question, logs, task_category, geneturing_data):
    """
    Score task classification for a single question.
    Returns: classification_score (0.0, 0.0000000001, or 1.0)
    """
    # Get expected task type
    expected_task_types = TASK_TYPE_MAPPING.get(task_category)
    if not expected_task_types:
        return 0.0  # Unknown task category
    
    # Extract predicted task type from logs
    predicted_task_type, classification_found, error_occurred = extract_task_classification_from_logs(logs)
    
    # Score the classification
    if error_occurred:
        return 0.0
    elif classification_found and predicted_task_type:
        if predicted_task_type in expected_task_types:
            return 1.0
        else:
            print(f"DEBUG: Question '{question}' misclassified as '{predicted_task_type}' instead of '{expected_task_types}'")
            return 0.0000000001  # Very small score for incorrect, better than erroring out. 
    else:
        return 0.0  # No classification found

def extract_parse_answer_from_doc_from_logs(logs):
    """
    Extract parse_answer_from_doc result from logs array.
    Returns: parsed_answer_result (error message or parsed answer)
    """
    if not logs or not isinstance(logs, list):
        return ""
    
    for log_entry in logs:
        if isinstance(log_entry, str):
            # Check for parse_answer_from_doc errors first
            if "parse_answer_from_doc: Error:" in log_entry:
                # Extract the error message after "Error: "
                error_match = re.search(r'parse_answer_from_doc: Error: (.+)', log_entry)
                if error_match:
                    return f"Error: {error_match.group(1)}"
            
            # Check for successful parsed answer
            elif "parse_answer_from_doc: Parsed answer:" in log_entry:
                # Extract the parsed answer
                answer_match = re.search(r"parse_answer_from_doc: Parsed answer: '([^']+)'", log_entry)
                if answer_match:
                    return answer_match.group(1)
                # Fallback for different quote formats
                answer_match = re.search(r'parse_answer_from_doc: Parsed answer: "([^"]+)"', log_entry)
                if answer_match:
                    return answer_match.group(1)
                # Fallback for no quotes
                answer_match = re.search(r'parse_answer_from_doc: Parsed answer: (.+)', log_entry)
                if answer_match:
                    return answer_match.group(1).strip()
            
            # Debug: Check if we have any parse_answer_from_doc entries at all
            elif "parse_answer_from_doc:" in log_entry:
                # Found a parse_answer_from_doc entry but couldn't match the pattern
                if DEBUG_PRINT:
                    print(f"DEBUG: Found parse_answer_from_doc entry but couldn't parse: {log_entry}")
    
    return ""

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

def extract_token_usage_info(logs):
    """
    Extract token usage information from the logs array.
    Returns: (total_tokens, input_tokens, output_tokens, total_cost, peak_context_usage, total_calls)
    """
    if not logs or not isinstance(logs, list):
        return "", "", "", "", "", ""
    
    total_tokens = ""
    input_tokens = ""
    output_tokens = ""
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
                # Parse both total and breakdown: "Total Tokens: 4,024 (3,877 input + 147 output)"
                total_match = re.search(r'Total Tokens:\s*([\d,]+)', log)
                if total_match:
                    total_tokens = int(total_match.group(1).replace(',', ''))
                
                # Extract input and output breakdown
                breakdown_match = re.search(r'\(([\d,]+)\s*input\s*\+\s*([\d,]+)\s*output\)', log)
                if breakdown_match:
                    input_tokens = int(breakdown_match.group(1).replace(',', ''))
                    output_tokens = int(breakdown_match.group(2).replace(',', ''))
                    
            elif log.startswith("Total Cost:"):
                match = re.search(r'Total Cost:\s*\$?([\d.]+)', log)
                if match:
                    total_cost = float(match.group(1))
            elif log.startswith("Peak Context Usage:"):
                match = re.search(r'Peak Context Usage:\s*([\d.]+)%', log)
                if match:
                    peak_context_usage = float(match.group(1))
    
    return total_tokens, input_tokens, output_tokens, total_cost, peak_context_usage, total_calls

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

def normalize_gene_symbol(gene_symbol):
    if not gene_symbol:
        return gene_symbol
    
    gene_symbol = str(gene_symbol).strip()
    # Remove <think>...</think> tags and content
    gene_symbol = re.sub(r'<think>.*?</think>', '', gene_symbol, flags=re.DOTALL)
    # Remove newlines and normalize spaces
    gene_symbol = re.sub(r'\s+', ' ', gene_symbol).strip()
    # Remove trailing periods and quotes
    gene_symbol = gene_symbol.rstrip('."\'')
    # Remove descriptive parts in parentheses: SYMBOL (Description) -> SYMBOL
    clean_symbol = re.sub(r'\s*\([^)]*\)', '', gene_symbol).strip()
    return clean_symbol

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
                'wild turkey': 'turkey',
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

    elif task == 'Gene SNP association':
        answer = answer.strip().replace('Answer: ', '')
        
        # Try specific patterns in order of confidence
        patterns = [
            r'associated with (?:the )?gene\s+([A-Z0-9_-]+)',  # "associated with the gene SYMBOL"
            r'is\s+[\'"]?([A-Z0-9_-]+)[\'"]?',                # "is 'SYMBOL'" or "is SYMBOL"
            r'^([A-Z0-9_-]+)\s*\([^)]*gene_id[^)]*\)',        # "SYMBOL (gene_id: XX)"
        ]
        
        # Try each pattern - if any matches, use it
        for pattern in patterns:
            match = re.search(pattern, answer, re.IGNORECASE)
            if match:
                answer = match.group(1)
                break
        # If no patterns match, keep original answer unchanged
        
        # Remove trailing punctuation only if we extracted a gene symbol
        if len(answer.split()) == 1 and answer.isupper():  # Looks like a gene symbol
            answer = re.sub(r'[.,:;!?]+$', '', answer)

    elif task == 'Gene alias':
        answer = answer.strip().replace('Answer: ', '')
        answer = normalize_gene_symbol(answer)

    elif task == 'Gene name conversion':
        answer = answer.strip().replace('Answer: ', '')
        answer = normalize_gene_symbol(answer)

    elif task == 'Protein-coding genes':
        answer = answer.strip().replace('Answer: ', '')
        
        # Case-insensitive matching, normalize to uppercase ground truth format
        answer_lower = answer.lower().strip()
        if answer_lower.startswith('yes') or answer_lower.startswith('true') or answer_lower == 'true':
            answer = 'TRUE'
        elif answer_lower.startswith('no') or answer_lower.startswith('false') or answer_lower == 'false':
            answer = 'NA'
        elif any(pattern in answer_lower for pattern in [
            'not a protein-coding gene',
            'is not protein-coding',
            'not protein-coding',
            'non-protein-coding',
            'non-coding'
        ]):
            answer = 'NA'

    elif task == 'Multi-species DNA aligment':
        answer = answer.strip().replace('Answer: ', '')
        
        # Remove markdown formatting and quotes
        answer = re.sub(r'\*\*(.*?)\*\*', r'\1', answer)
        answer = answer.strip('"\'')
        
        # Clean up extra spaces and newlines
        answer = re.sub(r'\s+', ' ', answer).strip()
        
        # Handle "No significant similarity found" cases first
        if 'no significant similarity found' in answer.lower():
            answer = "No significant similarity found"  # Normalize to exact ground truth format
        else:
            # Try to match species names (case-insensitive)
            for species_name, short_name in mapper.items():
                if answer.lower().startswith(species_name.lower()):
                    answer = short_name
                    break
            else:
                # If no exact species match, check if it starts with any short name
                for short_name in mapper.values():
                    if answer.lower().startswith(short_name):
                        answer = short_name
                        break
                else:
                    # Default: keep full organism name but make lowercase
                    answer = answer.lower()

    elif task == 'Human genome DNA aligment':
        answer = answer.strip().replace('Answer: ', '')
        # Remove markdown formatting if present
        answer = re.sub(r'\*\*([^*]+)\*\*', r'\1', answer)

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
    
    # Track task classification scores
    task_classification_scores = defaultdict(list)
    
    for task_file in glob.glob(os.path.join(folder, '*.json')):
        task = os.path.basename(task_file).replace('.json', '')
        if DEBUG_PRINT:
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

            entry_logs = []
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
                            entry_logs = entry[3]  # Store logs for classification scoring
                        else:
                            result['num_model_calls'] = ""
                            
                        # Add elapsed time (6th element) if it exists
                        if len(entry) > 5:
                            result['elapsed_time'] = entry[5]
                        else:
                            result['elapsed_time'] = ""
                        
                        # Extract token usage information from logs (4th element)
                        if len(entry) > 3 and isinstance(entry[3], list):
                            total_tokens, input_tokens, output_tokens, total_cost, peak_context_usage, total_calls = extract_token_usage_info(entry[3])
                            result['total_tokens'] = total_tokens
                            result['input_tokens'] = input_tokens  # NEW
                            result['output_tokens'] = output_tokens  # NEW
                            result['total_cost'] = total_cost
                            result['peak_context_usage'] = peak_context_usage
                            result['total_calls'] = total_calls
                            
                            # Extract stages completed information
                            result['stages_completed'] = extract_stages_completed(entry[3])
                        else:
                            result['total_tokens'] = ""
                            result['input_tokens'] = ""  # NEW
                            result['output_tokens'] = ""  # NEW
                            result['total_cost'] = ""
                            result['peak_context_usage'] = ""
                            result['total_calls'] = ""
                            result['stages_completed'] = ""
                            
                        break
                
                # Score task classification
                classification_score = score_task_classification_for_question(question, entry_logs, task, qas)
                result['task_classification_score'] = classification_score
                task_classification_scores[task].append(classification_score)

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
                result['input_tokens'] = ""
                result['output_tokens'] = ""
                result['total_cost'] = ""
                result['peak_context_usage'] = ""
                result['total_calls'] = ""
                result['stages_completed'] = ""
                result['task_classification_score'] = 0.0
                task_classification_scores[task].append(0.0)
                
                if should_exclude:
                    result['score'] = 'EXCLUDED'
                    result['success'] = 'EXCLUDED'
                    # Don't add to correct array
                else:
                    result['score'] = 0
                    result['success'] = 'Missing'
                    correct.append(0)
            
            # Extract parse_answer_from_doc result for all cases (answered or not)
            parsed_answer_result = extract_parse_answer_from_doc_from_logs(entry_logs)
            result['parse_answer_result'] = parsed_answer_result
            
            question_results.append(result)
        
        # Calculate task summary (excluding the excluded questions)
        total_score = sum(correct) / len(correct) if correct else 0
        effective_total = len(info) - excluded_count  # Total minus excluded
        
        if DEBUG_PRINT:
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
                    'api_urls', 'num_model_calls', 'elapsed_time', 
                    'total_tokens', 'total_cost', 'peak_context_usage', 'total_calls', 'stages_completed',
                    'task_classification_score', 'parse_answer_result', 'input_tokens', 'output_tokens']
        
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
    
    if DEBUG_PRINT:
        print(f"\n=== EVALUATION SUMMARY{output_suffix.upper()} ===")
        print(f"Overall Score: {overall_score:.4f}")
        print(f"Detailed reports saved to: {output_dir}")
        print(f"- Question-level details: {os.path.basename(details_file)}")
        print(f"- Task-level summary: {os.path.basename(summary_file)}")
    
    return task_summaries, overall_score

# Add these functions to evaluate.py

import csv
import os
import json
import re
from pathlib import Path

def is_task_classification_results(results_dir):
    """Check if results directory contains task classification results."""
    return "task_classification" in results_dir

def get_classification_output_files(results_dir):
    """Generate output file paths for task classification evaluation."""
    # Extract model name from path: .../gpt-5-nano_agent/task_classification/
    model_name = os.path.basename(os.path.dirname(results_dir))
    
    base_dir = "results/nba/evaluation_reports/task_classification/"
    os.makedirs(base_dir, exist_ok=True)
    
    details_file = f"{model_name}_details.csv"
    summary_file = f"{model_name}_summary.json"
    
    return os.path.join(base_dir, details_file), os.path.join(base_dir, summary_file)

def parse_classification_details_from_logs(logs):
    """Extract classification details from logs."""
    details = {
        'raw_response': '',
        'confidence': '',
        'reasoning': ''
    }
    
    if not logs:
        return details
    
    # Look for classification details section
    in_details_section = False
    for log in logs:
        if "TASK CLASSIFICATION DETAILS" in log:
            in_details_section = True
            continue
        
        if in_details_section:
            if log.startswith("Raw LLM Response:"):
                details['raw_response'] = log.replace("Raw LLM Response:", "").strip()
            elif log.startswith("Confidence:"):
                details['confidence'] = log.replace("Confidence:", "").strip()
            elif log.startswith("Reasoning:"):
                details['reasoning'] = log.replace("Reasoning:", "").strip()
            elif log.startswith("===") or log.startswith("Stage"):
                # End of details section
                break
    
    return details

def evaluate_task_classification(results_dir, qas_file="data/geneturing.json"):
    """
    Evaluate task classification results.
    Creates detailed CSV and summary JSON files.
    """
    print("RUNNING TASK CLASSIFICATION EVALUATION")
    print("=" * 50)
    
    # Get output file paths
    details_file, summary_file = get_classification_output_files(results_dir)
    print(f"Details output: {details_file}")
    print(f"Summary output: {summary_file}")
    
    # Load reference data
    with open(qas_file, 'r') as f:
        qas_data = json.load(f)
    
    # Collect all results
    all_results = []
    task_scores = {}  # Track scores per task category
    
    for task_name in qas_data.keys():
        task_file = os.path.join(results_dir, f"{task_name}.json")
        
        if not os.path.exists(task_file):
            print(f"Warning: {task_file} not found")
            continue
        
        with open(task_file, 'r') as f:
            task_results = json.load(f)
        
        # Initialize task score tracking
        if task_name not in task_scores:
            task_scores[task_name] = []
        
        for entry in task_results:
            if len(entry) < 6:
                print(f"Warning: Invalid entry format in {task_name}")
                continue
            
            question = entry[0]
            ground_truth = entry[1]  # Expected task type from main.py
            prediction = entry[2]    # Predicted task type from classification
            logs = entry[3]
            api_calls = entry[4]
            elapsed_time = entry[5]
            
            # Calculate score using existing function
            score = score_task_classification_for_question(question, logs, task_name, qas_data)
            success = score == 1.0
            
            # Extract predicted task type from logs for display
            predicted_task_type, _, _ = extract_task_classification_from_logs(logs)
            
            # Parse additional details from logs
            details = parse_classification_details_from_logs(logs)
            
            # Store result
            result_row = {
                'task': task_name,
                'question': question,
                'ground_truth': ground_truth,
                'prediction': predicted_task_type or prediction,  # Use extracted or fallback
                'score': score,
                'success': success,
                'elapsed_time': elapsed_time,
                'raw_classification_response': details['raw_response'],
                'confidence': details['confidence'],
                'reasoning': details['reasoning']
            }
            
            all_results.append(result_row)
            task_scores[task_name].append(score)
    
    # Write detailed CSV
    if all_results:
        with open(details_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['task', 'question', 'ground_truth', 'prediction', 
                         'score', 'success', 'elapsed_time', 'raw_classification_response', 
                         'confidence', 'reasoning']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        
        print(f"Detailed results written to: {details_file}")
    
    # Calculate summary statistics
    total_questions = len(all_results)
    total_correct = sum(1 for r in all_results if r['success'])
    overall_accuracy = total_correct / total_questions if total_questions > 0 else 0.0
    
    # Per-task accuracy
    task_accuracy = {}
    for task_name, scores in task_scores.items():
        if scores:
            accuracy = sum(scores) / len(scores)
            task_accuracy[task_name] = {
                'accuracy': accuracy,
                'correct': sum(scores),
                'total': len(scores)
            }
    
    # Calculate average elapsed time
    avg_elapsed_time = sum(r['elapsed_time'] for r in all_results) / total_questions if total_questions > 0 else 0.0
    
    # Create summary in same format as normal evaluation
    summary = {
        'overall_metrics': {
            'total_questions': total_questions,
            'total_correct': total_correct,
            'overall_accuracy': overall_accuracy,
            'average_elapsed_time': avg_elapsed_time
        },
        'task_breakdown': task_accuracy,
        'model_info': {
            'results_directory': results_dir,
            'evaluation_type': 'task_classification'
        }
    }
    
    # Write summary JSON
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary to console (same format as normal evaluation)
    print("=" * 50)
    print("TASK CLASSIFICATION EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Total Questions: {total_questions}")
    print(f"Correct Classifications: {total_correct}")
    print(f"Overall Accuracy: {overall_accuracy:.3f}")
    print(f"Average Time per Question: {avg_elapsed_time:.2f}s")
    print()
    print("Per-Task Breakdown:")
    print("-" * 50)
    for task_name, metrics in task_accuracy.items():
        print(f"{task_name}: {metrics['correct']}/{metrics['total']} ({metrics['accuracy']:.3f})")
    
    print(f"\nSummary written to: {summary_file}")
    return summary
        
def main_evaluation(folder_path, qas_file='data/geneturing.json', use_updated_logic=True, mode='both'):
    """
    Main evaluation function that can be called programmatically
    Args:
        folder_path: Path to the results folder containing JSON files
        qas_file: Path to the geneturing.json file (default: 'data/geneturing.json')
        use_updated_logic: Whether to use updated logic by default (default: True)
        mode: Evaluation mode - 'original', 'updated', or 'both' (default: 'both')
    """
    
    if is_task_classification_results(folder_path):
        return evaluate_task_classification(folder_path, qas_file)
    
    # Check if required files exist
    original_qas = qas_file
    updated_qas = qas_file.replace('.json', '_updated.json')
    
    if not os.path.exists(original_qas):
        print(f"Error: {original_qas} not found!")
        return

    original_results = None
    updated_results = None
    original_overall = None
    updated_overall = None

    # Run original evaluation if requested
    if mode in ['original', 'both']:
        print("RUNNING ORIGINAL EVALUATION")
        original_results, original_overall = run_evaluation(
            folder=folder_path,
            qas_file=original_qas,
            output_suffix="",
            use_updated_logic=False
        )

    # Run updated evaluation if requested and file exists
    if mode in ['updated', 'both'] and os.path.exists(updated_qas):
        print("\n\nRUNNING UPDATED EVALUATION")
        updated_results, updated_overall = run_evaluation(
            folder=folder_path,
            qas_file=updated_qas,
            output_suffix="_updated",
            use_updated_logic=use_updated_logic
        )
    elif mode in ['updated', 'both']:
        print(f"\nWarning: {updated_qas} not found. Cannot run updated evaluation.")

    # Run comparison only if both evaluations were completed
    if mode == 'both' and original_results is not None and updated_results is not None:
        # Compare results
        print(f'\n{"="*82}')
        print("COMPARISON SUMMARY")
        print(f'{"="*82}')
        print(f"{'Task#':<5} {'Task':<30} {'Original':<12} {'Updated':<12} {'Difference':<12}")
        print(f'{"-"*82}')

        # Use the same task ordering as the original QAS file (matches main.py iteration order)
        qas_data = json.load(open(qas_file))
        qas_task_order = list(qas_data.keys())

        # Get all tasks that exist in results, ordered by QAS file sequence
        available_tasks = set(original_results.keys()) if original_results else set(updated_results.keys())
        all_tasks = [task for task in qas_task_order if task in available_tasks]

        # Create task items with their indices (based on order in original_results)
        task_items = [(i, task) for i, task in enumerate(all_tasks)]

        for task_idx, task in task_items:
            orig_score = original_results.get(task, {}).get('score', 0.0)
            upd_score = updated_results.get(task, {}).get('score', 0.0)
            diff = upd_score - orig_score
            print(f"{task_idx:<5} {task:<30} {orig_score:<12.4f} {upd_score:<12.4f} {diff:+.4f}")
        
        print(f'{"-"*82}')
        print(f"{'':>5} {'OVERALL':<30} {original_overall:<12.4f} {updated_overall:<12.4f} {updated_overall-original_overall:+.4f}")
        print(f'{"="*82}')
        
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
    # Print single-mode summary when not running comparison
    elif mode in ['original', 'updated']:
        results = original_results if mode == 'original' else updated_results
        overall_score = original_overall if mode == 'original' else updated_overall
        
        if results is not None and overall_score is not None:
            print(f'\n{"="*60}')
            print(f"{mode.upper()} EVALUATION SUMMARY")
            print(f'{"="*60}')
            print(f"{'Task#':<5} {'Task':<35} {'Score':<10}")
            print(f'{"-"*60}')
            
            # Use the same task ordering as the original QAS file
            qas_data = json.load(open(qas_file))
            qas_task_order = list(qas_data.keys())
            
            # Get all tasks that exist in results, ordered by QAS file sequence
            available_tasks = set(results.keys())
            all_tasks = [task for task in qas_task_order if task in available_tasks]
            
            # Create task items with their indices
            task_items = [(i, task) for i, task in enumerate(all_tasks)]
            
            for task_idx, task in task_items:
                score = results.get(task, {}).get('score', 0.0)
                print(f"{task_idx:<5} {task:<35} {score:<10.4f}")
            
            print(f'{"-"*60}')
            print(f"{'':>5} {'OVERALL':<35} {overall_score:<10.4f}")
            print(f'{"="*60}')

if __name__ == '__main__':
    # Command line interface - result dir path to evaluate
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate GeneGPT results')
    parser.add_argument('folder_path', help='Path to results folder')
    parser.add_argument('--mode', choices=['original', 'updated', 'both'], 
                    default='updated', help='Evaluation mode: original, updated, or both (default: updated)')
    
    args = parser.parse_args()
    
    main_evaluation(args.folder_path, mode=args.mode)