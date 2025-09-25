import os
import sys
import json
import re
import traceback
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime

TEST_NUM_CHAR = 1000000  # 1 or 5 million characters for testing
# Add parent directory to path for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

# Import your existing functions - try multiple import paths
try:
    from nanobioagent.core.model_utils import create_langchain_model, invoke_llm_with_config, get_model_config
except ImportError:
    try:
        from ..nanobioagent.core.model_utils import create_langchain_model, invoke_llm_with_config, get_model_config
    except ImportError:
        # Add the nanobioagent directory to path
        nanobioagent_dir = parent_dir / "nanobioagent"
        sys.path.append(str(nanobioagent_dir))
        from ..nanobioagent.core.model_utils import create_langchain_model, invoke_llm_with_config, get_model_config

def get_all_model_names(
    config_file: str = "model_config.json",
    name_filter: Optional[str] = None,
    num_parameters_b_min = None,
    num_parameters_b_max = None
) -> List[str]:
    """
    Get model names from config file with optional filtering.
    
    Args:
        config_file: Path to the model config JSON file
        name_filter: Optional string to filter model names (case-insensitive substring match)
        num_parameters_b_min: Minimum number of parameters (in billions)
        num_parameters_b_max: Maximum number of parameters (in billions)
    
    Returns:
        List of filtered model names
    """
    # Try to find the config file in common locations
    config_path = None
    possible_paths = [
        Path(config_file),  # Direct path provided
        Path("config") / config_file,  # In config directory
        parent_dir / "config" / config_file,  # Relative to parent
        parent_dir / "nanobioagent" / "config" / config_file,  # In nanobioagent/config
    ]
    
    for path in possible_paths:
        if path.exists():
            config_path = path
            break
    
    if not config_path:
        raise FileNotFoundError(f"Could not find {config_file} in any of these locations: {possible_paths}")
    
    print(f"Loading model names from: {config_path}")
    
    # Load the config file
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    
    # Extract models data
    if "models" in config_data:
        models_data = config_data["models"]
    else:
        # If no "models" section, look for model names at top level
        exclude_keys = {"_defaults", "_metadata", "nvidia_nim_prefixes", "version", "description"}
        models_data = {key: value for key, value in config_data.items() if key not in exclude_keys}
    
    print(f"Found {len(models_data)} models in config file")
    
    def parse_parameter_count(model_name: str, model_config: dict) -> Optional[float]:
        """Parse parameter count from model config, return in billions"""
        
        # Check if there's explicit parameter info in the config
        if isinstance(model_config, dict) and 'num_parameters' in model_config:
            param_count = model_config['num_parameters']
            if isinstance(param_count, (int, float)):
                # Convert from actual parameter count to billions
                return param_count / 1_000_000_000
        
        # If no num_parameters field, return None
        return None
    
    # Apply filters
    filtered_models = []
    
    for model_name, model_config in models_data.items():
        # Apply name filter
        if name_filter and name_filter.lower() not in model_name.lower():
            continue
        
        # Apply parameter size filters
        if num_parameters_b_min is not None or num_parameters_b_max is not None:
            param_count_billions = parse_parameter_count(model_name, model_config)
            
            # Skip models without parameter info when parameter filtering is requested
            if param_count_billions is None:
                continue
            
            # Check parameter range
            if num_parameters_b_min is not None and param_count_billions < num_parameters_b_min:
                continue
            if num_parameters_b_max is not None and param_count_billions > num_parameters_b_max:
                continue
        
        filtered_models.append(model_name)
    
    # Print filtering results
    if name_filter or num_parameters_b_min is not None or num_parameters_b_max is not None:
        filters_applied = []
        if name_filter:
            filters_applied.append(f"name contains '{name_filter}'")
        if num_parameters_b_min is not None:
            filters_applied.append(f"params >= {num_parameters_b_min}B")
        if num_parameters_b_max is not None:
            filters_applied.append(f"params <= {num_parameters_b_max}B")
        
        print(f"Applied filters: {', '.join(filters_applied)}")
        print(f"Filtered to {len(filtered_models)} models")
    
    return filtered_models
    
def construct_test_prompt(num_char: int, output_filename: str, input_directory: str) -> Dict[str, Any]:
    """
    Build a test prompt of specified length using real genomics data from cache files.    
    Args:
        num_char: Target number of characters
        output_filename: Where to save the constructed prompt (JSON file)
        input_directory: Path to api_cache directory
    Returns:
        dict: {"prompt": text, "actual_length": int, "files_used": [list], "metadata": dict}
    """
    # Get all cache files sorted by modification time (oldest first)
    cache_dir = Path(input_directory)
    if not cache_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_directory}")
    
    # Get all files and sort by modification time (oldest first)
    cache_files = []
    for file_path in cache_dir.glob("*"):
        if file_path.is_file():
            cache_files.append((file_path.stat().st_mtime, file_path))
    
    cache_files.sort(key=lambda x: x[0])  # Sort by modification time
    print(f"Found {len(cache_files)} cache files")
    
    # Build the accumulated text
    accumulated_text = ""
    files_used = []
    
    for _, file_path in cache_files:
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().strip()
            
            # If we're getting close to target, check if adding this file would exceed it
            if len(accumulated_text) + len(content) > num_char:
                # Calculate how much we can still add
                remaining_chars = num_char - len(accumulated_text)
                if remaining_chars > 100:  # Only add if we have meaningful space left
                    # Add partial content
                    content = content[:remaining_chars-50]  # Leave some buffer
                    accumulated_text += "\n\n---\n\n" + content
                    files_used.append(str(file_path.name))
                break
            
            # Add the full content
            if accumulated_text:
                accumulated_text += "\n\n---\n\n"
            accumulated_text += content
            files_used.append(str(file_path.name))
            
            print(f"Added {file_path.name}, current length: {len(accumulated_text)} chars")
            
            # Check if we've reached our target
            if len(accumulated_text) >= num_char:
                break
                
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
    
    # Add the simple question at the end
    question = "\n\nRead the above genomics data and tell me the last word."
    full_prompt = accumulated_text + question
    
    # Extract the actual last word from the content for verification
    words = accumulated_text.strip().split()
    last_word = words[-1] if words else "N/A"
    
    # Create result dictionary
    result = {
        "prompt": full_prompt,
        "actual_length": len(full_prompt),
        "target_length": num_char,
        "files_used": files_used,
        "metadata": {
            "construction_date": datetime.now().isoformat(),
            "expected_last_word": last_word,
            "genomics_content_length": len(accumulated_text),
            "question_length": len(question),
            "total_files_available": len(cache_files),
            "files_actually_used": len(files_used)
        }
    }
    
    # Save to file
    output_path = Path(output_filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== SUMMARY ===")
    print(f"Target length: {num_char:,} chars")
    print(f"Actual length: {len(full_prompt):,} chars")
    print(f"Files used: {len(files_used)} of {len(cache_files)} available")
    print(f"Expected answer: '{last_word}'")
    print(f"Saved to: {output_filename}")
    
    return result


# Example usage function
def test_construct_prompt():
    """Test the construct_test_prompt function with different sizes."""
    # cache_dir = r"C:/Users/ssg_h/dev/nanobioagent/api_cache"
    cache_dir = r"api_cache"
    # Test different sizes
    # test_sizes = [10000, 50000, 100000, 500000, 1000000]
    test_sizes = [TEST_NUM_CHAR] # try 10 million first
    
    for size in test_sizes:
        print(f"\n{'='*50}")
        print(f"Constructing prompt with {size:,} characters")
        print(f"{'='*50}")
        
        output_file = f"tests/test_prompt_{size}.json"
        
        try:
            result = construct_test_prompt(size, output_file, cache_dir)
            
            # Quick preview
            prompt_preview = result["prompt"][:200] + "..."
            print(f"Prompt preview: {prompt_preview}")
            
        except Exception as e:
            print(f"Error constructing prompt of size {size}: {e}")
            
def test_model_context_token_info(model_name: str, prompt: str, num_char: int = -1) -> Dict[str, Any]:
    """
    Test a single model's context limits and measure actual token ratios.
    
    Args:
        model_name: Name of the model to test
        prompt: The prompt text to send
        num_char: If > 0, truncate prompt to this many characters (-1 = no truncation)
    
    Returns:
        dict: {
            "model_name": str,
            "success": bool,
            "prompt_length_chars": int,
            "tokens_used": int or None,  # Extracted from error message
            "token_limit": int or None,  # Extracted from error message  
            "char_to_token_ratio": float or None,  # prompt_length_chars / tokens_used
            "error_message": str or None,
            "response": str or None,  # If successful
            "model_config": dict  # From get_model_config
        }
    """
    
    # Truncate prompt if requested
    test_prompt = prompt[:num_char] if num_char > 0 else prompt
    prompt_length = len(test_prompt)
    
    # Get model config for reference
    model_config = get_model_config(model_name)
    
    result = {
        "model_name": model_name,
        "success": False,
        "prompt_length_chars": prompt_length,
        "tokens_used": None,
        "token_limit": None,
        "char_to_token_ratio": None,
        "error_message": None,
        "response": None,
        "model_config": model_config
    }
    
    try:
        print(f"Testing {model_name} with {prompt_length:,} characters...")
        
        # Create model and test
        llm = create_langchain_model(model_name)
        if not llm:
            result["error_message"] = "Failed to create model"
            return result
        
        # Try to invoke the model
        response = invoke_llm_with_config(llm, test_prompt, model_name)
        
        # If we get here, the request succeeded
        result["success"] = True
        result["response"] = response
        print(f"✅ {model_name}: SUCCESS - Got response")
        
    except Exception as e:
        error_str = str(e)
        result["error_message"] = error_str
        
        # Try to extract token information from error message
        tokens_used, token_limit = parse_token_error(error_str)
        
        if tokens_used:
            result["tokens_used"] = tokens_used
            result["char_to_token_ratio"] = prompt_length / tokens_used
            
        if token_limit:
            result["token_limit"] = token_limit
            
        if tokens_used and token_limit:
            print(f"❌ {model_name}: FAILED - Used {tokens_used} tokens, limit {token_limit}")
            print(f"   Ratio: {prompt_length / tokens_used:.2f} chars/token")
        else:
            print(f"❌ {model_name}: FAILED - {error_str[:100]}...")
    
    return result


def parse_token_error(error_message: str) -> tuple[Optional[int], Optional[int]]:
    """
    Extract token usage and limits from error messages.
    
    Returns:
        tuple: (tokens_used, token_limit) or (None, None) if not found
    """
    
    # Pattern 1: "prompt is [[5636]] long while only 4096 is supported" (NVIDIA NIM)
    pattern1 = r"prompt is \[\[(\d+)\]\] long while only (\d+) is supported"
    match = re.search(pattern1, error_message)
    if match:
        return int(match.group(1)), int(match.group(2))
    
    # Pattern 2: "Request too large: 5636 tokens > 4096 maximum" (OpenAI style)
    pattern2 = r"Request too large: (\d+) tokens > (\d+) maximum"
    match = re.search(pattern2, error_message)
    if match:
        return int(match.group(1)), int(match.group(2))
        
    # Pattern 3: "maximum context length is 4096 tokens, however you requested 5636" (OpenAI)
    pattern3 = r"maximum context length is (\d+) tokens, however you requested (\d+)"
    match = re.search(pattern3, error_message)
    if match:
        return int(match.group(2)), int(match.group(1))  # Note: swapped order
    
    # Pattern 4: Claude format - "prompt is too long: 217042 tokens > 200000 maximum"
    pattern4 = r"prompt is too long: (\d+) tokens > (\d+) maximum"
    match = re.search(pattern4, error_message)
    if match:
        return int(match.group(1)), int(match.group(2))
    
    # Pattern 5: "This model's maximum context length is 8192 tokens. However, your request has 416177 input tokens"
    pattern5 = r"maximum context length is (\d+) tokens.*?your request has (\d+) input tokens"
    match = re.search(pattern5, error_message)
    if match:
        return int(match.group(2)), int(match.group(1))  # tokens_used, token_limit

    # Pattern 6: "Prompt length (413955) exceeds maximum input length (127999)"
    pattern6 = r"Prompt length \((\d+)\) exceeds maximum input length \((\d+)\)"
    match = re.search(pattern6, error_message)
    if match:
        return int(match.group(1)), int(match.group(2))  # tokens_used, token_limit

    # Pattern 7: "max_tokens must be at least 1, got -151813" (indicates prompt too long)
    pattern7 = r"max_tokens must be at least 1, got (-?\d+)"
    match = re.search(pattern7, error_message)
    if match:
        negative_max_tokens = int(match.group(1))
        print(f"    Detected negative max_tokens: {negative_max_tokens} (prompt way too long)")
        return None, None
    
    # Pattern 8: OpenAI rate limit - "Limit 500000, Requested 1250032"
    pattern8 = r"Limit (\d+), Requested (\d+)"
    match = re.search(pattern8, error_message)
    if match:
        return int(match.group(2)), int(match.group(1))  # tokens_used, token_limit
    
    # Pattern 9: Gemini format - "input token count (2114730) exceeds the maximum number of tokens allowed (1048575)"
    pattern9 = r"input token count \((\d+)\) exceeds the maximum number of tokens allowed \((\d+)\)"
    match = re.search(pattern9, error_message)
    if match:
        return int(match.group(1)), int(match.group(2))  # tokens_used, token_limit
    
    # Pattern 10: Generic token count extraction
    # Look for any "X tokens" mentions
    token_numbers = re.findall(r'(\d+)\s*tokens?', error_message)
    if len(token_numbers) >= 2:
        # Assume first is usage, second is limit (or vice versa)
        nums = [int(x) for x in token_numbers]
        if len(nums) >= 2:
            # Heuristic: larger number is likely the usage, smaller is limit
            # But this might be wrong, so be careful
            return max(nums), min(nums)
    
    return None, None


def bulk_test_model_context_token_info(model_lists: List[str], prompt_file_name: str, num_char: int = -1) -> List[Dict[str, Any]]:
    """
    Test multiple models with the same prompt to measure token ratios.
    
    Args:
        model_lists: List of model names to test
        prompt_file_name: JSON file containing the prompt (from construct_test_prompt)
        num_char: If > 0, truncate prompt to this many characters (-1 = no truncation)
    
    Returns:
        List of results from test_model_context_token_info
    """
    
    # Load the prompt
    print(f"Loading prompt from {prompt_file_name}...")
    with open(prompt_file_name, 'r', encoding='utf-8') as f:
        prompt_data = json.load(f)
    
    prompt = prompt_data["prompt"]
    print(f"Loaded prompt: {len(prompt):,} characters")
    
    if num_char > 0:
        print(f"Will truncate to {num_char:,} characters for testing")
    
    # Test each model
    results = []
    
    print(f"\nTesting {len(model_lists)} models...")
    print("="*60)
    
    for i, model_name in enumerate(model_lists, 1):
        print(f"\n[{i}/{len(model_lists)}] Testing {model_name}")
        
        try:
            result = test_model_context_token_info(model_name, prompt, num_char)
            results.append(result)
        except Exception as e:
            print(f"❌ CRITICAL ERROR testing {model_name}: {e}")
            # Add error result
            results.append({
                "model_name": model_name,
                "success": False,
                "prompt_length_chars": len(prompt[:num_char] if num_char > 0 else prompt),
                "tokens_used": None,
                "token_limit": None,
                "char_to_token_ratio": None,
                "error_message": f"Critical error: {str(e)}",
                "response": None,
                "model_config": {}
            })
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY RESULTS")
    print("="*60)
    
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    with_ratios = [r for r in results if r["char_to_token_ratio"] is not None]
    
    print(f"Total models tested: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"With token ratios: {len(with_ratios)}")
    
    if with_ratios:
        print(f"\nToken ratios (chars/token):")
        for r in sorted(with_ratios, key=lambda x: x["char_to_token_ratio"] or 0):
            ratio = r["char_to_token_ratio"]
            tokens = r["tokens_used"]
            limit = r["token_limit"]
            config_context = r["model_config"].get("context_window", "Unknown")
            
            # Compare tested limit vs config context window
            context_match = "✅" if limit == config_context else "❌"
            
            print(f"  {r['model_name']:<40} {ratio:.2f} chars/token ({tokens} used, {limit} limit) Config: {config_context} {context_match}")
    
    # Also show models that failed but we know their limits
    failed_with_limits = [r for r in failed if r["token_limit"] is not None]
    if failed_with_limits:
        print(f"\nFailed models with detected limits:")
        for r in failed_with_limits:
            limit = r["token_limit"]
            config_context = r["model_config"].get("context_window", "Unknown")
            context_match = "✅" if limit == config_context else "❌"
            print(f"  {r['model_name']:<40} Failed - Limit: {limit}, Config: {config_context} {context_match}")
    
    return results


# Convenience function to save results
def save_token_test_results(results: List[Dict[str, Any]], output_filename: str):
    """Save token test results to JSON file."""
    
    output_data = {
        "test_date": datetime.now().isoformat(),
        "total_models": len(results),
        "successful_models": len([r for r in results if r["success"]]),
        "models_with_ratios": len([r for r in results if r["char_to_token_ratio"] is not None]),
        "results": results
    }
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {output_filename}")


def test_all_models():
    # Step 1: Create a large prompt (if not already done)
    # construct_test_prompt(1000000, "large_prompt.json", r"C:\Users\ssg_h\dev\nanobioagent\api_cache")
    
    # Step 2: Test a single model
    # result = test_model_context_token_info("google/gemma-2-9b-it", "large_prompt.json", 100000)
    
    # Step 3: Test multiple models
    all_models = get_all_model_names()
    print(f"Testing all {len(all_models)} models from config...")
    
    models_to_test = all_models.copy()  
    # Filter to specific types or exclude some
    # models_to_test = [m for m in models_to_test if "nvidia" in m or "google" in m]
    models_to_test = [m for m in models_to_test if not m.startswith("ollama")]
    models_to_test = [m for m in models_to_test if not m.startswith("text-davinci")]
    models_to_test = [m for m in models_to_test if not m.startswith("huggingface")]
    # or just override the list completely
    '''
    models_to_test = [
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "gpt-4.1",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite"
    ]
    '''
    results = bulk_test_model_context_token_info(
        models_to_test,
        f"tests/test_prompt_{TEST_NUM_CHAR}.json", 
        num_char=-1  # do all
    )
    
    # Step 4: Save results
    save_token_test_results(results, f"tests/token_ratio_results_{TEST_NUM_CHAR}.json")


if __name__ == "__main__":
    # test_construct_prompt()
    test_all_models()