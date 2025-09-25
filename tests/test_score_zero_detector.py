#!/usr/bin/env python3
"""
Test script to detect cases where scoring gives 0 but the answer is inside the prediction.
Works with evaluation report CSV files that already contain scores.
Usage: python test_score_zero_detector.py [csv_folder_path] [file_filter]
"""

import pandas as pd
import glob
import os
import sys
import json
import re
from typing import List, Dict, Any
from collections import defaultdict

def check_answer_in_prediction(prediction: str, ground_truth: str, task: str) -> bool:
    """
    Check if the ground truth answer is contained within the prediction text.
    
    Args:
        prediction: The model's prediction text
        ground_truth: The correct answer
        task: The task type for context-specific checking
        
    Returns:
        bool: True if answer is found in prediction, False otherwise
    """
    if pd.isna(prediction) or pd.isna(ground_truth) or not prediction or not ground_truth:
        return False
    
    # Convert both to strings and normalize
    pred_text = str(prediction).lower().strip()
    answer_text = str(ground_truth).lower().strip()
    
    # Skip if they are the same (shouldn't happen for score=0 cases, but just in case)
    if pred_text == answer_text:
        return True
    
    # Direct substring check
    if answer_text in pred_text:
        return True
    
    # For gene-related tasks, also check without case sensitivity and common variations
    if 'gene' in task.lower() or 'snp' in task.lower():
        # Remove common prefixes/suffixes and check
        answer_clean = re.sub(r'^(rs|gene|snp)', '', answer_text).strip()
        if answer_clean and answer_clean in pred_text:
            return True
            
        # Check for gene symbol patterns (e.g., LINC01270)
        if re.search(rf'\b{re.escape(answer_text)}\b', pred_text):
            return True
    
    # Check if answer appears in quoted text or after common phrases
    patterns = [
        rf'"{re.escape(answer_text)}"',  # Quoted
        rf'is {re.escape(answer_text)}',  # "is ANSWER"
        rf'gene {re.escape(answer_text)}',  # "gene ANSWER"
        rf'{re.escape(answer_text)}\.',  # "ANSWER."
        rf'{re.escape(answer_text)}\s',  # "ANSWER "
    ]
    
    for pattern in patterns:
        if re.search(pattern, pred_text):
            return True
    
    return False

def get_csv_files(csv_folder: str, file_filter: str = "*_details_updated.csv") -> List[str]:
    """
    Get CSV files based on filter pattern.
    
    Args:
        csv_folder: Path to folder containing CSV files
        file_filter: Filter pattern - can be:
                   - "*_details_updated.csv" (default)
                   - "*_details_*.csv" 
                   - specific filename like "ibm_granite-3.3-8b-instruct_agent_details_updated.csv"
                   - "*" for all CSV files
    
    Returns:
        List of matching CSV file paths
    """
    if file_filter == "*":
        pattern = "*.csv"
    elif not file_filter.endswith('.csv'):
        # If it's just a pattern like "*_details_updated", add .csv
        pattern = file_filter + ".csv" if not file_filter.endswith('*') else file_filter + ".csv"
    else:
        pattern = file_filter
    
    csv_files = glob.glob(os.path.join(csv_folder, pattern))
    
    # If it's a specific filename and not found, try without path
    if not csv_files and not ('*' in file_filter):
        specific_file = os.path.join(csv_folder, file_filter)
        if os.path.exists(specific_file):
            csv_files = [specific_file]
    
    return sorted(csv_files)

def analyze_csv_files(csv_folder: str, file_filter: str = "*_details_updated.csv") -> Dict[str, Any]:
    """
    Analyze CSV files in evaluation_reports folder to find zero score cases with answers inside.
    
    Args:
        csv_folder: Path to folder containing evaluation report CSV files
        file_filter: Filter pattern for which files to process
        
    Returns:
        Dictionary containing analysis results
    """
    
    results = {
        'total_files': 0,
        'total_rows': 0,
        'zero_score_cases': 0,
        'zero_score_with_answer_inside': 0,
        'problematic_cases': [],
        'task_breakdown': defaultdict(lambda: {
            'total': 0, 
            'zero_scores': 0, 
            'zero_with_answer': 0
        }),
        'files_processed': [],
        'file_filter': file_filter
    }
    
    # Find CSV files based on filter
    csv_files = get_csv_files(csv_folder, file_filter)
    
    if not csv_files:
        print(f"No CSV files found matching pattern '{file_filter}' in: {csv_folder}")
        print("Available CSV files in directory:")
        all_csvs = glob.glob(os.path.join(csv_folder, "*.csv"))
        for csv_file in all_csvs[:10]:  # Show first 10
            print(f"  - {os.path.basename(csv_file)}")
        if len(all_csvs) > 10:
            print(f"  ... and {len(all_csvs) - 10} more")
        return results
    
    print(f"Found {len(csv_files)} files matching pattern '{file_filter}':")
    
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        print(f"Processing: {filename}")
        results['files_processed'].append(filename)
        results['total_files'] += 1
        
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file)
            results['total_rows'] += len(df)
            
            # Check required columns exist
            required_cols = ['score', 'ground_truth', 'prediction', 'task']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"  Warning: Missing columns {missing_cols} in {filename}")
                continue
            
            # Debug: Check score column values and types
            print(f"  Score column type: {df['score'].dtype}")
            print(f"  Unique score values: {sorted(df['score'].unique())}")
            print(f"  Score value counts: {df['score'].value_counts().head()}")
            
            # Filter for score = 0 cases (handle both string and numeric)
            zero_score_df = df[(df['score'] == 0) | (df['score'] == '0') | (df['score'] == 0.0)].copy()
            
            if len(zero_score_df) == 0:
                print(f"  No zero score cases found in {filename}")
                continue
                
            print(f"  Found {len(zero_score_df)} zero score cases")
            
            # Check each zero score case
            for idx, row in zero_score_df.iterrows():
                task = row['task']
                ground_truth = row['ground_truth']
                prediction = row['prediction']
                question = row.get('question', 'N/A')
                
                results['zero_score_cases'] += 1
                results['task_breakdown'][task]['total'] += 1
                results['task_breakdown'][task]['zero_scores'] += 1
                
                # Check if answer is inside prediction
                answer_inside = check_answer_in_prediction(prediction, ground_truth, task)
                
                if answer_inside:
                    results['zero_score_with_answer_inside'] += 1
                    results['task_breakdown'][task]['zero_with_answer'] += 1
                    
                    # Store problematic case
                    case = {
                        'file': filename,
                        'task': task,
                        'question': question,
                        'ground_truth': ground_truth,
                        'prediction': prediction,
                        'score': row['score'],
                        'success': row.get('success', 'N/A')
                    }
                    
                    # Add any additional columns for context
                    for col in ['api_urls', 'elapsed_time', 'total_tokens', 'normalized_answer', 'normalized_prediction']:
                        if col in row:
                            case[col] = row[col]
                    
                    results['problematic_cases'].append(case)
        
        except Exception as e:
            print(f"  Error processing {filename}: {str(e)}")
            continue
    
    return results

def print_analysis_summary(results: Dict[str, Any]):
    """Print a summary of the analysis results."""
    
    print("\n" + "="*70)
    print("ZERO SCORE ANALYSIS SUMMARY")
    print("="*70)
    
    print(f"File filter used: {results['file_filter']}")
    print(f"Files processed: {results['total_files']}")
    print(f"Total rows analyzed: {results['total_rows']}")
    print(f"Zero score cases: {results['zero_score_cases']}")
    print(f"Zero score cases with answer inside prediction: {results['zero_score_with_answer_inside']}")
    
    if results['zero_score_cases'] > 0:
        percentage = (results['zero_score_with_answer_inside'] / results['zero_score_cases']) * 100
        print(f"Percentage of zero scores that contain the answer: {percentage:.2f}%")
    
    print(f"\nFiles processed:")
    for filename in results['files_processed']:
        print(f"  - {filename}")
    
    print("\n" + "-"*50)
    print("BREAKDOWN BY TASK")
    print("-"*50)
    
    # Sort tasks by number of problematic cases
    sorted_tasks = sorted(results['task_breakdown'].items(), 
                         key=lambda x: x[1]['zero_with_answer'], reverse=True)
    
    for task, stats in sorted_tasks:
        if stats['zero_scores'] > 0:
            contained_pct = (stats['zero_with_answer'] / stats['zero_scores']) * 100
            
            print(f"\n{task}:")
            print(f"  Zero scores: {stats['zero_scores']}")
            print(f"  Zero scores with answer inside: {stats['zero_with_answer']} ({contained_pct:.1f}%)")

def save_problematic_cases(results: Dict[str, Any], output_file: str):
    """Save problematic cases to a JSON file for detailed review."""
    
    with open(output_file, 'w') as f:
        json.dump({
            'summary': {
                'total_files': results['total_files'],
                'total_rows': results['total_rows'],
                'zero_score_cases': results['zero_score_cases'],
                'zero_score_with_answer_inside': results['zero_score_with_answer_inside'],
                'files_processed': results['files_processed']
            },
            'task_breakdown': dict(results['task_breakdown']),
            'problematic_cases': results['problematic_cases']
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_file}")

def save_problematic_cases_csv(results: Dict[str, Any], output_file: str):
    """Save problematic cases to a CSV file for easy review."""
    
    if not results['problematic_cases']:
        print("No problematic cases to save.")
        return
    
    df = pd.DataFrame(results['problematic_cases'])
    df.to_csv(output_file, index=False)
    print(f"Problematic cases CSV saved to: {output_file}")

def print_sample_cases(results: Dict[str, Any], num_samples: int = 5):
    """Print sample problematic cases."""
    
    if not results['problematic_cases']:
        print("No problematic cases found.")
        return
    
    print(f"\n" + "-"*70)
    print(f"SAMPLE PROBLEMATIC CASES (showing first {min(num_samples, len(results['problematic_cases']))})")
    print("-"*70)
    
    for i, case in enumerate(results['problematic_cases'][:num_samples]):
        print(f"\n--- Example {i+1} ---")
        print(f"File: {case['file']}")
        print(f"Task: {case['task']}")
        print(f"Question: {case['question']}")
        print(f"Ground Truth: {case['ground_truth']}")
        print(f"Prediction: {case['prediction'][:300]}{'...' if len(str(case['prediction'])) > 300 else ''}")
        print(f"Score: {case['score']}")

def get_default_results_path():
    """
    Get the default results path based on current working directory.
    Works whether running from nanobioagent/ or nanobioagent/tests/
    """
    current_dir = os.getcwd()
    
    # If we're in tests/ directory, go up one level
    if current_dir.endswith('tests'):
        base_dir = os.path.dirname(current_dir)
    # If we're already in nanobioagent/ or similar
    else:
        base_dir = current_dir
    
    # Look for results folder
    results_path = os.path.join(base_dir, "results", "clean", "evaluation_reports")
    
    # If that doesn't exist, try alternative paths
    if not os.path.exists(results_path):
        # Try just results/evaluation_reports
        alt_path = os.path.join(base_dir, "results", "evaluation_reports")
        if os.path.exists(alt_path):
            return alt_path
        
        # Try clean/evaluation_reports in current dir
        alt_path = os.path.join(current_dir, "clean", "evaluation_reports")
        if os.path.exists(alt_path):
            return alt_path
    
    return results_path

def main():
    """Main function to run when script is executed directly."""
    
    # Get CSV folder path and file filter
    if len(sys.argv) >= 2:
        csv_folder = sys.argv[1]
    else:
        # Use relative path based on current directory
        csv_folder = get_default_results_path()
        print(f"No folder specified. Using default: {csv_folder}")
    
    if len(sys.argv) >= 3:
        file_filter = sys.argv[2]
    else:
        # Default filter - details_updated files
        file_filter = "*_details_updated.csv"
        print(f"No file filter specified. Using default: {file_filter}")
    
    print(f"Current working directory: {os.getcwd()}")
    print(f"Analyzing CSV files in: {csv_folder}")
    print(f"File filter: {file_filter}")
    
    # Check if folder exists
    if not os.path.exists(csv_folder):
        print(f"Error: Folder not found: {csv_folder}")
        print("\nLooking for evaluation_reports folders...")
        
        # Help user find the right path
        current_dir = os.getcwd()
        possible_paths = [
            os.path.join(current_dir, "results", "clean", "evaluation_reports"),
            os.path.join(current_dir, "results", "evaluation_reports"),
            os.path.join(current_dir, "clean", "evaluation_reports"),
            os.path.join(current_dir, "evaluation_reports"),
        ]
        
        if current_dir.endswith('tests'):
            parent_dir = os.path.dirname(current_dir)
            possible_paths.extend([
                os.path.join(parent_dir, "results", "clean", "evaluation_reports"),
                os.path.join(parent_dir, "results", "evaluation_reports"),
            ])
        
        found_paths = [path for path in possible_paths if os.path.exists(path)]
        
        if found_paths:
            print("Found these evaluation_reports folders:")
            for path in found_paths:
                print(f"  {path}")
            print(f"\nTry running with: python test_score_zero_detector.py \"{found_paths[0]}\"")
        else:
            print("No evaluation_reports folders found in expected locations.")
            print("Please specify the full path to your evaluation_reports folder.")
        
        return
    
    print("\n" + "="*70)
    print("STARTING ZERO SCORE ANALYSIS")
    print("="*70)
    
    # Run the analysis
    results = analyze_csv_files(csv_folder, file_filter)
    
    if results['total_files'] == 0:
        print("No CSV files were processed.")
        return
    
    # Print summary
    print_analysis_summary(results)
    
    # Print sample cases
    print_sample_cases(results)
    
    # Save results
    if results['problematic_cases']:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        json_output = f"problematic_cases_{timestamp}.json"
        csv_output = f"problematic_cases_{timestamp}.csv"
        
        save_problematic_cases(results, json_output)
        save_problematic_cases_csv(results, csv_output)
        
        print(f"\nSUMMARY:")
        print(f"Found {results['zero_score_with_answer_inside']} cases where score=0 but answer is in prediction!")
        print(f"This represents {results['zero_score_with_answer_inside']/results['zero_score_cases']*100:.1f}% of all zero score cases.")
    else:
        print("\nNo problematic cases found - your scoring seems to be working correctly!")

# Convenience functions for interactive use
def analyze_folder(csv_folder: str, file_filter: str = "*_details_updated.csv"):
    """Convenience function for interactive analysis."""
    return analyze_csv_files(csv_folder, file_filter)

def quick_analysis(file_filter: str = "*_details_updated.csv"):
    """Quick analysis with default path."""
    return analyze_csv_files(get_default_results_path(), file_filter)

if __name__ == "__main__":
    main()