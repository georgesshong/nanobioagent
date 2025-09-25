"""
Data utilities for GeneTuring evaluation - handles input data and output results.

This module provides functions to:
1. Access GeneTuring questions by index
2. Find question indices from text 
3. Filter and analyze evaluation results
"""

import json
import os
import pandas as pd
import re
from typing import List, Dict, Union, Tuple, Optional, Any
from pathlib import Path

def resolve_project_path(relative_path):
    """Convert relative path to absolute path from project root."""
    project_root = Path(__file__).parent.parent.parent  # Go up to nanobioagent/ root
    return str(project_root / relative_path)

def get_question_text_from_idx(question_idx: str, data_files: str = 'data/geneturing.json', 
                            return_metadata: bool = False) -> Union[str, Dict[str, Any], None]:
    """
    Get question text from index like '0|34' (task 0, question 34).
    
    Args:
        question_idx: Index in format "task_num|question_num" (e.g., "0|34")
        data_files: Path to GeneTuring JSON file
        return_metadata: If True, return dict with question, task, ground_truth, etc.
        
    Returns:
        Question text string, or dict with metadata if return_metadata=True, or None if not found
        
    Examples:
        >>> get_question_text_from_idx("0|34")
        "What is the official gene symbol of LMP10?"
        
        >>> get_question_text_from_idx("0|34", return_metadata=True)
        {"question": "What is the official gene symbol of LMP10?", 
        "task": "Gene alias", "ground_truth": "PSMB10"}
    """
    try:
        # Parse the index
        if '|' not in question_idx:
            raise ValueError(f"Invalid question_idx format. Expected 'task|question', got '{question_idx}'")
        
        task_idx, q_idx = question_idx.split('|')
        task_idx, q_idx = int(task_idx), int(q_idx)
        
        data_files = resolve_project_path(data_files)
        
        # Load the data
        with open(data_files, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert dict structure to list for indexing
        task_names = list(data.keys())
        
        # Validate indices
        if task_idx >= len(task_names):
            raise IndexError(f"Task index {task_idx} out of range (0-{len(task_names)-1})")
        
        task_name = task_names[task_idx]
        questions = list(data[task_name].keys())
        
        if q_idx >= len(questions):
            raise IndexError(f"Question index {q_idx} out of range for task {task_idx} (0-{len(questions)-1})")
        
        question = questions[q_idx]
        ground_truth = data[task_name][question]
        
        if return_metadata:
            return {
                'question': question,
                'task': task_name,
                'ground_truth': ground_truth,
                'task_idx': task_idx,
                'question_idx': q_idx
            }
        else:
            return question
            
    except Exception as e:
        print(f"Error getting question from idx '{question_idx}': {e}")
        return None

def get_question_idx_from_text(question_text: str, data_files: str = 'data/geneturing.json',
                            method: str = "simple", embedding_model: str = "all-MiniLM-L6-v2",
                            threshold: float = 0.8, return_confidence: bool = True) -> Union[str, Tuple[str, float], Tuple[None, float]]:
    """
    Find question index from text with confidence scoring.
    
    Args:
        question_text: The question text to search for
        data_files: Path to GeneTuring JSON file  
        method: Matching method - "simple", "tfidf", or "embedding"
        embedding_model: Model name for embedding method
        threshold: Minimum confidence threshold (0.0-1.0)
        return_confidence: If True, return (idx, confidence), else just idx
        
    Returns:
        If return_confidence=False: "0|34" or None if no match above threshold
        If return_confidence=True: ("0|34", 0.95) or (None, 0.0)
        
    Examples:
        >>> get_question_idx_from_text("What is the official gene symbol of LMP10?")
        ("0|34", 1.0)
        
        >>> get_question_idx_from_text("some random text", threshold=0.9)
        (None, 0.0)
    """
    try:
        # Load the data
        data_files = resolve_project_path(data_files)
        with open(data_files, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        best_match_idx = None
        best_confidence = 0.0
        
        # Search through all questions
        for task_idx, (task_name, questions_dict) in enumerate(data.items()):
            for q_idx, (question, ground_truth) in enumerate(questions_dict.items()):
                
                if method == "simple":
                    confidence = _simple_text_similarity(question_text, question)
                elif method == "tfidf":
                    confidence = _tfidf_similarity(question_text, question)
                elif method == "embedding":
                    confidence = _embedding_similarity(question_text, question, embedding_model)
                else:
                    raise ValueError(f"Unknown method: {method}. Use 'simple', 'tfidf', or 'embedding'")
                
                if confidence > best_confidence and confidence >= threshold:
                    best_confidence = confidence
                    best_match_idx = f"{task_idx}|{q_idx}"
        
        if return_confidence:
            return (best_match_idx, best_confidence)
        else:
            return best_match_idx
            
    except Exception as e:
        print(f"Error finding question idx for text: {e}")
        if return_confidence:
            return (None, 0.0)
        else:
            return None


def get_data_from_results(result_file: str, 
                        filter_score: str = "[0,1]",
                        filter_success: Optional[List[str]] = None,
                        filter_task: Optional[List[str]] = None,
                        dataType: str = "question",
                        return_format: str = "list") -> Union[List, pd.DataFrame]:
    """
    Extract and filter data from evaluation results CSV.
    
    Args:
        result_file: Path to evaluation results CSV
        filter_score: Score filter - supports ranges "[0.5,1]", "(0,0.5)", comparisons ">0.8", "<0.5", "==1", "!=0"
        filter_success: List of success values to include (e.g., ["YES", "NO"])
        filter_task: List of task names to include (e.g., ["Gene alias", "SNP location"])
        dataType: Comma-separated fields to return (e.g., "question,task,score,idx")
        return_format: "list" for list of dicts/values, "df" for pandas DataFrame
        
    Returns:
        Filtered data as list or DataFrame based on return_format
        
    Examples:
        >>> # Get failed questions with metadata
        >>> get_data_from_results("results.csv", filter_score="<0.5", 
        ...                      filter_success=["NO"], dataType="question,task,score,idx")
        [{"question": "What is...", "task": "Gene alias", "score": 0.0, "idx": "0|34"}, ...]
        
        >>> # Get just question texts as simple list
        >>> get_data_from_results("results.csv", filter_score=">0.8", dataType="question")
        ["What is the official gene symbol...", "Which chromosome is...", ...]
    """
    try:
        result_file = resolve_project_path(result_file)
        # Load the results CSV
        df = pd.read_csv(result_file)
        
        # Apply filters
        filtered_df = df.copy()
        
        # Score filtering
        if filter_score != "[0,1]":
            score_mask = _parse_score_filter(filtered_df['score'], filter_score)
            filtered_df = filtered_df[score_mask]
        
        # Success filtering
        if filter_success is not None:
            filtered_df = filtered_df[filtered_df['success'].isin(filter_success)]
        
        # Task filtering
        if filter_task is not None:
            filtered_df = filtered_df[filtered_df['task'].isin(filter_task)]
        
        # Parse dataType and extract columns
        fields = [field.strip() for field in dataType.split(',')]
        
        # Handle special 'idx' field
        if 'idx' in fields:
            # Add idx column by converting questions to indices
            # print("Converting questions to indices (this may take a moment)...")
            idx_column = []
            for question in filtered_df['question']:
                idx, _ = get_question_idx_from_text(question, method="simple", return_confidence=True)
                idx_column.append(idx)
            filtered_df = filtered_df.copy()  # Avoid SettingWithCopyWarning
            filtered_df['idx'] = idx_column
        
        # Extract requested fields
        available_fields = [f for f in fields if f in filtered_df.columns]
        missing_fields = [f for f in fields if f not in filtered_df.columns]
        
        if missing_fields:
            print(f"Warning: Fields not found in data: {missing_fields}")
        
        result_df = filtered_df[available_fields]
        
        # Return in requested format
        if return_format == "df":
            return result_df
        elif return_format == "list":
            if len(available_fields) == 1:
                # Single field - return simple list of values
                return result_df.iloc[:, 0].tolist()
            else:
                # Multiple fields - return list of dicts
                return result_df.to_dict('records')
        else:
            raise ValueError(f"Unknown return_format: {return_format}. Use 'list' or 'df'")
            
    except Exception as e:
        print(f"Error processing results file '{result_file}': {e}")
        if return_format == "df":
            return pd.DataFrame()
        else:
            return []


# Helper functions for text similarity

def _simple_text_similarity(text1: str, text2: str) -> float:
    """Simple exact and fuzzy text matching."""
    # Exact match
    if text1.strip().lower() == text2.strip().lower():
        return 1.0
    
    # Basic fuzzy matching - count common words
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if len(words1) == 0 and len(words2) == 0:
        return 1.0
    if len(words1) == 0 or len(words2) == 0:
        return 0.0
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0


def _tfidf_similarity(text1: str, text2: str) -> float:
    """TF-IDF based similarity (requires scikit-learn)."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(similarity)
        
    except ImportError:
        print("Warning: scikit-learn not available, falling back to simple similarity")
        return _simple_text_similarity(text1, text2)


def _embedding_similarity(text1: str, text2: str, model_name: str) -> float:
    """Embedding-based similarity (requires sentence-transformers)."""
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        model = SentenceTransformer(model_name)
        embeddings = model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)
        
    except ImportError:
        print("Warning: sentence-transformers not available, falling back to simple similarity")
        return _simple_text_similarity(text1, text2)

def _parse_score_filter(scores: pd.Series, filter_str: str) -> pd.Series:
    """Parse score filter string and return boolean mask."""
    filter_str = filter_str.strip()
    
    # Convert scores to numeric, replacing non-numeric with NaN
    numeric_scores = pd.to_numeric(scores, errors='coerce')
    
    # Handle comparison operators
    comparison_pattern = r'^(>=|<=|>|<|==|!=)\s*([+-]?\d*\.?\d+)$'
    match = re.match(comparison_pattern, filter_str)
    
    if match:
        operator, value = match.groups()
        value = float(value)
        
        if operator == '>':
            return numeric_scores > value
        elif operator == '>=':
            return numeric_scores >= value
        elif operator == '<':
            return (numeric_scores < value) & numeric_scores.notna()
        elif operator == '<=':
            return (numeric_scores <= value) & numeric_scores.notna()
        elif operator == '==':
            return numeric_scores == value
        elif operator == '!=':
            return (numeric_scores != value) & numeric_scores.notna()
    
    # Handle range notation
    range_pattern = r'^([\[\(])\s*([+-]?\d*\.?\d+)\s*,\s*([+-]?\d*\.?\d+)\s*([\]\)])$'
    match = re.match(range_pattern, filter_str)
    
    if match:
        left_bracket, min_val, max_val, right_bracket = match.groups()
        min_val, max_val = float(min_val), float(max_val)
        
        if left_bracket == '[':
            min_condition = numeric_scores >= min_val
        else:  # '('
            min_condition = numeric_scores > min_val
            
        if right_bracket == ']':
            max_condition = numeric_scores <= max_val
        else:  # ')'
            max_condition = numeric_scores < max_val
            
        return min_condition & max_condition & numeric_scores.notna()
    
    raise ValueError(f"Invalid score filter format: '{filter_str}'. "
                    f"Use comparisons like '>0.5' or ranges like '[0.5,1]'")


# Utility functions for common use cases

def parse_task_indices(task_idx_string: str) -> List[Tuple[int, int]]:
    """
    Parse task indices string into list of tuples.
    
    Args:
        task_idx_string: String like "0|34,4|6,2|15"
        
    Returns:
        List of (task_num, question_num) tuples: [(0,34), (4,6), (2,15)]
        
    Example:
        >>> parse_task_indices("0|34,4|6")
        [(0, 34), (4, 6)]
    """
    if not task_idx_string.strip():
        return []
    
    indices = []
    for idx_pair in task_idx_string.split(','):
        idx_pair = idx_pair.strip()
        if '|' not in idx_pair:
            raise ValueError(f"Invalid index format: '{idx_pair}'. Expected 'task|question'")
        
        task_idx, q_idx = idx_pair.split('|')
        indices.append((int(task_idx), int(q_idx)))
    
    return indices


def format_task_idx(task_num: int, question_num: int) -> str:
    """
    Format task and question numbers into index string.
    
    Args:
        task_num: Task index (0-based)
        question_num: Question index within task (0-based)
        
    Returns:
        Formatted index string like "0|34"
        
    Example:
        >>> format_task_idx(0, 34)
        "0|34"
    """
    return f"{task_num}|{question_num}"


def get_failure_summary(result_file: str) -> Dict[str, Any]:
    """
    Get summary statistics of failures by task, score ranges, etc.
    
    Args:
        result_file: Path to evaluation results CSV
        
    Returns:
        Dictionary with failure statistics
        
    Example:
        >>> summary = get_failure_summary("results.csv")
        >>> print(f"Total failures: {summary['total_failures']}")
        >>> print(f"Failures by task: {summary['failures_by_task']}")
    """
    try:
        df = pd.read_csv(result_file)
        
        total_questions = len(df)
        failures = df[df['success'] == 'NO']
        total_failures = len(failures)
        
        # Failures by task
        failures_by_task = failures['task'].value_counts().to_dict()
        
        # Score distribution
        score_ranges = {
            'perfect (1.0)': len(df[df['score'] == 1.0]),
            'high (0.8-0.99)': len(df[(df['score'] >= 0.8) & (df['score'] < 1.0)]),
            'medium (0.5-0.79)': len(df[(df['score'] >= 0.5) & (df['score'] < 0.8)]),
            'low (0.1-0.49)': len(df[(df['score'] >= 0.1) & (df['score'] < 0.5)]),
            'zero (0.0)': len(df[df['score'] == 0.0]),
            'negative': len(df[df['score'] < 0.0])
        }
        
        return {
            'total_questions': total_questions,
            'total_failures': total_failures,
            'failure_rate': total_failures / total_questions if total_questions > 0 else 0,
            'failures_by_task': failures_by_task,
            'score_distribution': score_ranges,
            'average_score': df['score'].mean(),
            'median_score': df['score'].median()
        }
        
    except Exception as e:
        print(f"Error generating failure summary: {e}")
        return {}

def list_available_results(results_dir: str = "results/nba/evaluation_reports/", 
                        file_suffix = None, sort_by: str = "name") -> List[str]:
    """
    List all available CSV result files in the directory.
    
    Args:
        results_dir: Directory containing evaluation results (relative to project root)
        file_suffix: Optional suffix to filter files (e.g., "_details_updated.csv")
        sort_by: Sort method - "name" (alphabetical), "mtime" (modification time), "size"
        
    Returns:
        List of CSV filenames matching the criteria, sorted as requested
    """
    try:
        # Resolve path relative to project root
        full_results_dir = resolve_project_path(results_dir)
        
        if not os.path.exists(full_results_dir):
            print(f"Directory not found: {full_results_dir}")
            return []
        
        # Get all CSV files
        all_csv_files = [f for f in os.listdir(full_results_dir) if f.endswith('.csv')]
        
        # Filter by suffix if provided
        if file_suffix:
            csv_files = [f for f in all_csv_files if f.endswith(file_suffix)]
            print(f"Found {len(csv_files)} files ending with '{file_suffix}' in {results_dir}:")
        else:
            csv_files = all_csv_files
            print(f"Found {len(csv_files)} CSV files in {results_dir}:")
        
        # Sort files based on criteria
        if sort_by == "name":
            sorted_files = sorted(csv_files)
        elif sort_by == "mtime":
            # Sort by modification time (oldest first)
            sorted_files = sorted(csv_files, 
                                key=lambda f: os.path.getmtime(os.path.join(full_results_dir, f)))
        elif sort_by == "size":
            # Sort by file size (smallest first)
            sorted_files = sorted(csv_files, 
                                key=lambda f: os.path.getsize(os.path.join(full_results_dir, f)))
        else:
            print(f"Unknown sort method '{sort_by}', using alphabetical")
            sorted_files = sorted(csv_files)
        
        # Display with timestamps if sorted by mtime
        if sort_by == "mtime":
            import datetime
            for f in sorted_files:
                mtime = os.path.getmtime(os.path.join(full_results_dir, f))
                timestamp = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
                print(f"  - {f} ({timestamp})")
        else:
            for f in sorted_files:
                print(f"  - {f}")
        
        return sorted_files
    
    except Exception as e:
        print(f"Error listing files: {e}")
        return []
    
if __name__ == "__main__":
    # Example usage and testing
    print("Testing data_utils functions...")
    
    # First, let's check if the data file exists and what it contains
    import os
    data_file = 'data/geneturing.json'
    
    print(f"Checking for data file: {data_file}")
    if os.path.exists(data_file):
        print("✓ Data file found")
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Debug the data structure  
            print(f"✓ Data type: {type(data)}")
            print(f"✓ Task names: {list(data.keys())}")
            
            if len(data) > 0:
                first_task_name = list(data.keys())[0]
                first_task_data = data[first_task_name]
                print(f"✓ First task: '{first_task_name}' with {len(first_task_data)} questions")
                
                # Test get_question_text_from_idx with valid data
                if len(first_task_data) > 0:
                    question = get_question_text_from_idx("0|0", data_file, return_metadata=True)
                    if question:
                        print(f"✓ Sample question: {question['question'][:100]}...")
                        print(f"✓ Task: {question['task']}")
                        print(f"✓ Ground truth: {question['ground_truth']}")
                        
                        # Test get_question_idx_from_text  
                        idx, confidence = get_question_idx_from_text(question['question'], data_file)
                        print(f"✓ Found index: {idx} with confidence: {confidence}")
                    else:
                        print("✗ Failed to get question from idx 0|0")
                else:
                    print("✗ No questions found in first task")
            else:
                print("✗ No tasks found in data file")
                
        except Exception as e:
            print(f"✗ Error loading data file: {e}")
    else:
        print(f"✗ Data file not found at: {os.path.abspath(data_file)}")
        print("Available files in data/ directory:")
        if os.path.exists('data'):
            for file in os.listdir('data'):
                print(f"  - {file}")
        else:
            print("  No data/ directory found")
    
    print("\ndata_utils module ready for use!")