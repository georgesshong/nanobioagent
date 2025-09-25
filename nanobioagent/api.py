"""
API module for NanoBioAgent
Contains the main gene_answer functions moved from main.py
"""
import io
import os
import re
import sys
import time
from contextlib import contextmanager
from .tools import ncbi_query as ncbi
from .tools import gene_utils as utils
from .tools.gene_utils import call_api, log_event, resolve_multiple_paths, resolve_single_path
from .core.agent_framework import answer_agent
from .core.prompts import get_prompt_header, get_prompt_direct, extract_answer
from .core.model_utils import UniversalCostTracker, call_llm

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

DEBUG_PRINT = True
# GeneGPT method configuration constants
SKIP_REPEAT_PROMPT = True
MAX_NUM_CALLS = 10
DEFAULT_MODEL_NAME = "gpt-4o-mini" # Default model for LLM calls in this module

# Ask a question to the common wrapper function that handles different methods and models
def gene_answer(question, answer=None, method='genegpt', **kwargs):
    """Main dispatcher - backward compatible"""
    handlers = {
        'direct': gene_answer_direct,
        'retrieve': geneGPT_answer_retrieve,  
        'code': gene_answer_code,          
        'agent': gene_answer_agent,
        'genegpt': gene_answer_genegpt
    }
    
    if method not in handlers:
        raise ValueError(f"Unknown method '{method}'. Available methods: {list(handlers.keys())}")
    
    if answer is None and method != 'retrieve':
        answer = geneGPT_answer_retrieve(question, method="simple")[2]
    
    if DEBUG_PRINT:
        log_event(f"Using method: {method}")
        
    return handlers[method](question, answer, **kwargs)

# A simple wrapper for sending question directly to the LLM without API calls.
def gene_answer_direct(question, answer=None, model_name=None, config_data=None, use_fallback=False, **kwargs):
    """
    Handle direct LLM calls without API interactions.
    Uses the model's internal knowledge only.
    """
    log_event(question)
    # Start timing
    question_start_time = time.time()
    # Set default answer if not provided
    if answer is None:
        answer = ""
    
    # Generate direct prompt
    direct_prompt = get_prompt_direct(question)
    
    # Call the model directly
    text = call_llm(direct_prompt, model_name, config_data, use_fallback)
    
    # Extract just the answer
    text = extract_answer(text)
    
    # Calculate timing
    elapsed_time = round(time.time() - question_start_time, 2)
    log_event(f"Question answered directly in {elapsed_time} seconds")

    # Return in standard format
    return [question, answer, text, [[direct_prompt, text]], [], elapsed_time]

# A simple wrapper for agentic framework approach
def gene_answer_agent(question, answer=None, model_name=None, config_data=None, config_dir=None, **kwargs):
    """
    Handle agent framework approach.
    Direct integration with the NBA agent framework
    """
    log_event(question)
    
    # Set default answer if not provided
    if answer is None:
        answer = ""
    
    # Set default model name
    if model_name is None:
        model_name = DEFAULT_MODEL_NAME
    
    # Call the agent framework (it handles config_dir=None internally)
    try:
        result = answer_agent(
            question=question,
            answer=answer,
            model_name=model_name,
            config_data=config_data,
            config_dir=config_dir,
            **kwargs
        )
        
        # Ensure result is in expected format
        if not isinstance(result, list) or len(result) != 6:
            log_event("Format error: Invalid result format from agent framework")
            return [
                question,
                answer,
                "Format error",
                ["Error: Invalid result format from agent framework"],
                [],
                0.0
            ]
        
        return result
        
    except Exception as e:
        log_event(f"Agent framework error: {str(e)}")
        return [
            question,
            answer,
            f"Agent error: {str(e)}",
            [f"Error in agent framework: {str(e)}"],
            [],
            0.0
        ]

# A simple wrapper for GeneGPT API-calling approach
def gene_answer_genegpt(question, answer=None, prompt=None, model_name=None, config_data=None, 
                        str_mask='111111', config_path=None, use_fallback=False, 
                        repeat_prompt=None, **kwargs):
    """
    Handle original GeneGPT API-calling approach.
    Taken from the GeneGPT codebase with minor modifications to enable calling alternative LLMs 
    """
    log_event(question)
    
    # Set default answer if not provided
    if answer is None:
        answer = ""
    
    # Start timing
    question_start_time = time.time()
    cost_tracker = UniversalCostTracker()
    # Generate prompt if not provided
    if prompt is None:
        mask = [bool(int(x)) for x in str_mask]
        prompt = get_prompt_header(mask)
    
    # Set default model name
    if model_name is None:
        model_name = DEFAULT_MODEL_NAME
    # Construct the full prompt
    q_prompt = prompt + f'Question: {question}\n'
    if repeat_prompt is None:
        repeat_prompt = q_prompt
    
    # Initialize tracking variables
    prompts = []
    api_urls = []
    num_calls = 0
    
    # Main API calling loop as per inference algorithm in paper
    while True:
        if num_calls >= MAX_NUM_CALLS:
            elapsed_time = round(time.time() - question_start_time, 2)
            log_event(f"Question timed out after {elapsed_time} seconds")
            return [question, answer, 'numError', prompts, api_urls, elapsed_time]
        
        # Add final call reminder if needed
        if num_calls == (MAX_NUM_CALLS - 2):
            reminder = "\nThis is your last opportunity to provide a final answer based on the information already collected. Please provide your best answer now using the format 'Answer: [your answer]'.\n"
            q_prompt += reminder
            if DEBUG_PRINT: 
                log_event("Adding final call reminder to prompt")
        
        # Rate limiting
        time.sleep(0.3)
        
        # Call the LLM
        text = call_llm(q_prompt, model_name, config_data, use_fallback, cost_tracker=cost_tracker)
        if DEBUG_PRINT:
            log_event(f"=== Model call #: {num_calls+1} ===")
            log_event(f"=== Model response start: ===\n{text}")
            log_event("=== Model response end ===")
        
        num_calls += 1
        if SKIP_REPEAT_PROMPT:
            prompts.append([q_prompt.replace(repeat_prompt, "(...)"), text])
        else:
            prompts.append([q_prompt, text])
        
        # Check for API URLs in response
        url_regex = r'\[(https?://[^\[\]]+)\]'
        text_str = str(text) if text is not None else ""
        matches = re.findall(url_regex, text_str)
        
        if matches:
            url = matches[0]
            api_urls.append(url)
            
            if DEBUG_PRINT:
                log_event(f"URL matches found ({len(matches)}) start: {matches}")
                log_event("URL matches found end")
            
            try:
                call = call_api(url)
                
                if 'blast' in url and 'CMD=Put' in url:
                    rid_match = re.search('RID = (.*)\n', call.decode('utf-8', errors='replace'))
                    if rid_match:
                        rid = rid_match.group(1)
                        call = rid
                
                if DEBUG_PRINT:
                    log_event(f"=== Web API result start: ===\n{call}")
                    log_event("=== Web API result end ===")
                
                if len(call) > 20000:
                    log_event(f'call too long, cut {len(call)} to 20000')
                    call = call[:20000]
                
                q_prompt = f'{q_prompt}{text}->[{call}]\n'
                
            except Exception as e:
                log_event(f"Error in API call: {str(e)}")
                elapsed_time = round(time.time() - question_start_time, 2)
                return [question, answer, f"Error: {str(e)}", prompts, api_urls, elapsed_time]
        else:
            # No more API calls - check for final answer
            text_str = str(text) if text is not None else ""
            if ('\n\nAnswer' in text_str) or (text_str.strip().startswith("Answer:")):
                text = extract_answer(text_str)
                if DEBUG_PRINT:
                    log_event(f"=== Extracted found Answer 2 start: ===\n{text}")
                    log_event("=== Extracted found Answer 2 end ===")
            
            elapsed_time = round(time.time() - question_start_time, 2)
            log_event(f"Question answered in {elapsed_time} seconds")
            log_event(f"=== Total calls #: {num_calls} ===")
            cost_summary = cost_tracker.get_output_summary()
            logs = prompts + [""] + cost_summary
            return [question, answer, text, logs, api_urls, elapsed_time]

# A simple wrapper for pure code-based function call approach
def gene_answer_code(question, answer=None, **kwargs):
    question_start_time = time.time()
    
    result = ncbi.gene_query(question)
    if answer is None:
        answer = result[0]
        
    elapsed_time = round(time.time() - question_start_time, 2)
    log_event(f"Question answered with Code approach in {elapsed_time} seconds")
    # question, answer, "", prompts, api_urls, elapsed_time
    # Return a simplified result structure with empty prompts and API URLs
    return [question, answer, result[0], [result[1]], [], elapsed_time]

# A simple wrapper for retieving answers from answer text files
def geneGPT_answer_retrieve(
    question,
    answer=None, 
    json_paths=['data/geneturing.json', 'data/genehop.json'],
    method="tfidf",  # Similarity method - "embedding", "tfidf", or "simple"
    embedding_model="all-MiniLM-L6-v2", 
    similarity_threshold=0.7,  # Default to 0 to always return best match
    cache_dir="data/cache",
    force_rebuild=False,
    verbose=False,
    **kwargs
):
    """
    Answer a genomic question by retrieving similar questions from a database and
    returning the corresponding answer, without any fallback to other methods.
    
    Args:
        question (str): The genomic question to answer
        json_paths (str or list): Path(s) to JSON file(s) containing QA pairs
        method (str): Similarity method - "embedding", "tfidf", or "simple"
        embedding_model (str): Name of the embedding model (for "embedding" method)
        similarity_threshold (float): Minimum similarity score to use retrieved answer (if no match above threshold, returns empty answer)
        cache_dir (str): Directory to store cached embeddings and retrieval systems
        force_rebuild (bool): If True, rebuilds the retrieval system even if cache exists
        verbose (bool): If True, prints status messages
        
    Returns:
        List containing [question, answer, text, prompts, api_urls, elapsed_time]
    """
    # Safety check and path resolution
    json_paths = resolve_multiple_paths(json_paths, ['data/geneturing.json', 'data/genehop.json'])
    cache_dir = resolve_single_path(cache_dir, "data/cache")

    # Start timing
    start_time = time.time()
    
    # Build or load the retrieval system
    qa_dict, retriever = utils.build_qa_retrieval_from_json(
        json_paths=json_paths,
        method=method,
        embedding_model=embedding_model,
        cache_dir=cache_dir,
        force_rebuild=force_rebuild,
        verbose=verbose
    )
    
    # Find the most similar question and its answer
    similar_question, answer, score = retriever(question)
    
    # Empty placeholders for compatibility with geneGPT_answer format
    api_urls = []
    
    # If we found a sufficiently similar question
    if score >= similarity_threshold:
        if verbose:
            log_event(f"Found similar question: {similar_question}")
            log_event(f"Similarity score: {score:.4f}")
            log_event(f"Retrieved answer: {answer}")
        
        # Add basic prompt info for the retrieved answer
        prompts = [[f"Retrieved similar question: {similar_question} (score: {score:.4f})", f"Answer: {answer}"]]
        
        # Calculate elapsed time
        elapsed_time = round(time.time() - start_time, 2)
        
        # Return in same format as geneGPT_answer for consistency
        return [question, answer, answer, prompts, api_urls, elapsed_time]
    
    # No similar question found above threshold
    if verbose:
        log_event(f"No similar question found above threshold ({score:.4f} < {similarity_threshold})")
    
    # Calculate elapsed time
    elapsed_time = round(time.time() - start_time, 2)
    
    # Return empty answer with the best match information
    prompts = [[f"Best match: {similar_question} (score: {score:.4f}, below threshold)", f"No answer returned"]]
    return [question, answer, "", prompts, api_urls, elapsed_time]

# helper function to compare different models and methods for a given question
def gene_compare(
    question,
    list_model_name=["gpt-4.1-mini"],
    list_method=["direct", "genegpt", "agent"],
    config_data=None, 
    str_mask='111111'
):
    """
    Compare answers to a genomic question across multiple models and methods.
    
    Args:
        question (str): The genomic question to answer
        list_model_name (list): List of model names to try
        list_method (list): List of methods to try
        json_paths (str or list): Path(s) to JSON file(s) for retrieval
        embedding_model (str): Name of the embedding model for retrieval
        similarity_threshold (float): Minimum similarity score for retrieval
        cache_dir (str): Directory for caching retrieval systems
        force_rebuild (bool): Whether to rebuild retrieval cache
        verbose (bool): Whether to print detailed information
        **kwargs: Additional arguments to pass to geneGPT_answer
        
    Returns:
        list: List of dictionaries containing comparison results
    """
    import pandas as pd
    import time
    from tqdm.auto import tqdm
    
    # Create result storage
    results = []

    retrieve_result = geneGPT_answer_retrieve(
        question=question
    )
    
    # Extract relevant information
    _, _, retrieve_answer, _, _, retrieve_time = retrieve_result
    
    # Add to results
    results.append({
        "model_name": "N/A",
        "method": "retrieve",
        "answer": retrieve_answer,
        "elapsed_time": retrieve_time
    })
    
    # Create all combinations of models and methods
    # combinations = [(model, method) for model in list_model_name for method in list_method]
    combinations = [(method, model) for method in list_method for model in list_model_name]
    
    # Run all combinations with a progress bar
    # for model, method in tqdm(combinations, desc="Running model combinations"):
    for method, model in tqdm(combinations, desc="Running model combinations"):
        if DEBUG_PRINT:
            log_event(f"\nRunning model: {model}, method: {method}")
        
        # Run the model with the given method
        try:
            
            # Call function with correct parameters
            start_time = time.time()
            result = gene_answer(question, retrieve_answer, model_name=model, method=method,
                                        config_data=config_data, str_mask=str_mask)
                
            # Extract relevant information
            if isinstance(result, list) and len(result) >= 3:
                answer = result[2]
                if len(result) >= 6:
                    elapsed_time = result[5]
                else:
                    elapsed_time = round(time.time() - start_time, 2)
            else:
                answer = str(result)
                elapsed_time = round(time.time() - start_time, 2)
            
            # Add to results
            results.append({
                "model_name": model,
                "method": method,
                "answer": answer,
                "elapsed_time": elapsed_time
            })
            
        except Exception as e:
            if DEBUG_PRINT:
                log_event(f"Error with model {model}, method {method}: {str(e)}")
            
            # Add error result
            results.append({
                "model_name": model,
                "method": method,
                "answer": f"ERROR: {str(e)}",
                "elapsed_time": -1
            })
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results)
    
    if DEBUG_PRINT:
        log_event("\nComparison Results:")
        log_event(df)
    
    return results

@contextmanager
def suppress_stdout():
    """Context manager to temporarily suppress stdout output."""
    original_stdout = sys.stdout
    sys.stdout = io.StringIO()  # Redirect stdout to a string buffer
    try:
        yield
    finally:
        sys.stdout = original_stdout  # Restore original stdout

# stop annoying prints when calling in notebooks
def gene_compare_df(question, list_model_name=[DEFAULT_MODEL_NAME], list_method=["direct", "genegpt"], config_data=None, str_mask='111111'):
    with suppress_stdout():
        import pandas as pd
        df = pd.DataFrame(gene_compare(question, list_model_name, list_method, config_data, str_mask))
    return df