import os
import json
import pickle
import hashlib
import time
import urllib.error
import urllib.request
import re
import importlib
import logging
from logging.handlers import RotatingFileHandler
from typing import Dict, List, Tuple, Callable, Optional, Any, Union
from sentence_transformers import SentenceTransformer

# Enable/Disable logging
ENABLE_FILE_LOGGING = True
ENABLE_CONSOLE_PRINTING = True

# Setup logger
logger = logging.getLogger("gene_logger")
logger.setLevel(logging.DEBUG)

# Default formatter with timestamp
default_formatter = logging.Formatter('%(asctime)s|%(levelname)s|%(message)s')
# Formatter without timestamp
simple_formatter = logging.Formatter('%(levelname)s|%(message)s')
# Formatter with only message
message_only_formatter = logging.Formatter('%(message)s')

# Console handler
if ENABLE_CONSOLE_PRINTING:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(default_formatter)
    logger.addHandler(console_handler)

# File handler
if ENABLE_FILE_LOGGING:
    file_handler = RotatingFileHandler("gene_tool.log", maxBytes=5*1024*1024, backupCount=3)
    file_handler.setFormatter(default_formatter)
    logger.addHandler(file_handler)

# Unified logging function, level can be "info", "debug", "warning", "error", "critical"
def log_event(message, level="info", enable_time_logging=False):
    if not level:
        for handler in logger.handlers:
            handler.setFormatter(message_only_formatter)
    elif not enable_time_logging:
        for handler in logger.handlers:
            handler.setFormatter(simple_formatter)
    else:
        for handler in logger.handlers:
            handler.setFormatter(default_formatter)
    getattr(logger, level.lower() or "info")(message)

# Function to call NCBI API with Caching
def call_api(url, wait_time=1, use_cache=True, cache_blast=True, cache_dir="api_cache"):
    """
    Makes an API call with caching support, including special handling for BLAST calls.
    """
    # Replace spaces with '+' for URL encoding
    url = url.replace(' ', '+')
    log_event(url)
    cache_dir = resolve_single_path(None, 'api_cache')
    
    # Create cache directory if it doesn't exist
    if use_cache or cache_blast:
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(os.path.join(cache_dir, "blast_mappings"), exist_ok=True)
    
    # Check if this is a BLAST GET request
    is_blast_get = 'blast.ncbi.nlm.nih.gov' in url and 'CMD=Get' in url
    
    # Get cache key for all requests
    cache_key = hashlib.md5(url.encode('utf-8')).hexdigest()
    cache_path = os.path.join(cache_dir, f"{cache_key}.cache")
    
    # Check if the response is cached (for any request type)
    if (use_cache or (is_blast_get and cache_blast)) and os.path.exists(cache_path):
        # print(f"TEMP 0: Using cache_path: {cache_path}")
        log_event(f"Using cached response for {url}")
        with open(cache_path, 'rb') as f:
            return f.read()
    
    # Not in cache - make the actual API call
    time.sleep(wait_time)
    
    # Handle BLAST PUT requests specially
    if 'blast.ncbi.nlm.nih.gov' in url and 'CMD=Put' in url:
        # print(f"TEMP 0: make_blast_put_request url: {url}")
        response = make_blast_put_request(url, cache_dir)
        
        # Cache the PUT response if enabled
        if use_cache or cache_blast:
            with open(cache_path, 'wb') as f:
                f.write(response)
        
        return response
    
    # Make regular API call for all other requests
    req = urllib.request.Request(url) 
    try:
        with urllib.request.urlopen(req) as response:
            call = response.read()
        
        # For BLAST GET requests, wait for results to be ready if not cached
        if is_blast_get:
            # Check if results are ready
            status_match = re.search(r'Status=(\w+)', call.decode('utf-8', errors='replace'))
            if status_match and status_match.group(1) == 'WAITING':
                log_event("BLAST results not ready yet, waiting 30 seconds...")
                time.sleep(30)
                # Try again after waiting
                return call_api(url, wait_time, use_cache, cache_blast, cache_dir)
        
        # Save to cache if enabled
        should_cache = use_cache or (is_blast_get and cache_blast)
        if should_cache:
            with open(cache_path, 'wb') as f:
                f.write(call)
        
        return call
    except urllib.error.HTTPError as e:
        log_event(f"HTTP Error {e.code}: {e.reason} for URL: {url}")
        error_response = f"API_ERROR:{e.code}:{e.reason}".encode('utf-8')
        return error_response
    except urllib.error.URLError as e:
        log_event(f"URL Error: {e.reason} for URL: {url}")
        error_response = f"API_ERROR:URLError:{e.reason}".encode('utf-8')
        return error_response
    except Exception as e:
        log_event(f"Unexpected error for URL {url}: {str(e)}")
        error_response = f"API_ERROR:Unexpected:{str(e)}".encode('utf-8')
        return error_response

def make_blast_put_request(url, cache_dir=None):
    """
    Handle BLAST PUT requests specifically, including sequence-based caching.
    """
    # Resolve cache directory
    if cache_dir is None:
        cache_dir = resolve_single_path(None, 'api_cache')
    
    # Extract the query sequence
    query_match = re.search(r'QUERY=([^&]+)', url)
    if not query_match:
        log_event("Could not extract query from BLAST URL")
        # Just make the regular API call without special handling
        req = urllib.request.Request(url) 
        with urllib.request.urlopen(req) as response:
            return response.read()
    
    # Get the sequence and create a hash for caching
    sequence = query_match.group(1)
    sequence_hash = hashlib.md5(sequence.encode('utf-8')).hexdigest()
    sequence_cache_file = os.path.join(cache_dir, "blast_mappings", f"{sequence_hash}.json")
    
    # Create blast_mappings directory if it doesn't exist
    os.makedirs(os.path.join(cache_dir, "blast_mappings"), exist_ok=True)
    # print(f"TEMP: Using cache directory: {cache_dir}")
    # print(f"TEMP: Using sequence_cache_file: {sequence_cache_file}")
    # Check if we already have results for this exact sequence
    if os.path.exists(sequence_cache_file):
        with open(sequence_cache_file, 'r') as f:
            try:
                mapping = json.load(f)
                if 'rid' in mapping and 'response' in mapping and 'timestamp' in mapping:
                    # Check if the cached RID is still valid (less than 23 hours old)
                    if time.time() - mapping['timestamp'] < 23 * 3600:
                        log_event(f"Using cached BLAST sequence results for: {sequence_hash}")
                        return mapping['response'].encode('utf-8')
            except json.JSONDecodeError:
                # If the cache file is corrupted, continue with the API call
                pass
    
    # Make the actual API call
    req = urllib.request.Request(url) 
    with urllib.request.urlopen(req) as response:
        response_data = response.read()
    
    # Extract RID from response
    response_text = response_data.decode('utf-8', errors='replace')
    rid_match = re.search('RID = (.*)\n', response_text)
    
    if rid_match:
        rid = rid_match.group(1)
        
        # Save mapping of sequence to RID for future use
        mapping = {
            'sequence': sequence,
            'rid': rid,
            'timestamp': time.time(),
            'response': response_text
        }
        
        with open(sequence_cache_file, 'w') as f:
            json.dump(mapping, f)
    
    return response_data

def handle_blast_get(url, wait_time, cache_blast, cache_dir=None, max_retries=10):
    """Handle BLAST GET requests with special caching logic and retry limit."""
    # Resolve cache directory
    cache_dir = resolve_single_path(cache_dir, 'api_cache')
    
    # Extract the RID from the URL
    rid_match = re.search(r'RID=([^&]+)', url)
    if not rid_match:
        log_event("Could not extract RID from BLAST GET URL")
        return make_api_call(url, wait_time, cache_dir)
    
    rid = rid_match.group(1)
    blast_result_file = os.path.join(cache_dir, f"blast_result_{rid}.cache")
    
    # Check if we have cached results for this RID
    if cache_blast and os.path.exists(blast_result_file):
        log_event(f"Using cached BLAST results for RID: {rid}")
        with open(blast_result_file, 'rb') as f:
            return f.read()
    
    # Initialize response to None
    response = None
    last_exception = None
    
    # Try up to max_retries times
    for attempt in range(max_retries):
        try:
            response = make_api_call(url, wait_time, cache_dir)
            
            # Check if results are ready (not containing "Status=WAITING")
            response_text = response.decode('utf-8', errors='ignore')
            if "Status=WAITING" in response_text or "BLAST results not ready" in response_text:
                if attempt < max_retries - 1:  # Not the last attempt
                    log_event(f"BLAST results not ready yet, waiting 30 seconds... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(30)
                    continue
                else:
                    log_event(f"BLAST results still not ready after {max_retries} attempts ({max_retries * 30} seconds). Giving up.")
                    # Results not ready, but we have a response - break to use it
                    break
            else:
                # Results are ready!
                log_event(f"BLAST results ready after {attempt + 1} attempt(s)")
                break
                
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                log_event(f"BLAST request failed (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(30)
                continue
            else:
                log_event(f"BLAST request failed after {max_retries} attempts: {e}")
                # Don't raise here - handle it after the loop
                break
    
    # Check if we have a valid response
    if response is None:
        # All attempts failed, raise the last exception
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError("BLAST request failed: No response received and no exception recorded")
    
    # Cache the results if enabled
    if cache_blast:
        with open(blast_result_file, 'wb') as f:
            f.write(response)
    
    return response

def make_api_call(url, wait_time, cache_dir=None):
    """Make the actual API call without caching."""
    # Resolve cache directory (though this function doesn't use it for caching)
    cache_dir = resolve_single_path(cache_dir, 'api_cache')
    
    time.sleep(wait_time)
    
    req = urllib.request.Request(url) 
    try:
        with urllib.request.urlopen(req) as response:
            return response.read()
    except urllib.error.HTTPError as e:
        log_event(f"HTTP Error {e.code}: {e.reason} for URL: {url}")
        return f"API_ERROR:{e.code}:{e.reason}".encode('utf-8')
    except urllib.error.URLError as e:
        log_event(f"URL Error: {e.reason} for URL: {url}")
        return f"API_ERROR:URLError:{e.reason}".encode('utf-8')
    except Exception as e:
        log_event(f"Unexpected error for URL {url}: {str(e)}")
        return f"API_ERROR:Unexpected:{str(e)}".encode('utf-8')

class EmbeddingRetriever:
    """Class-based retriever using sentence embeddings for similarity."""
    
    def __init__(self, questions, answers, model_name="all-MiniLM-L6-v2"):
        """Initialize with questions, answers and embedding model."""
        from sentence_transformers import SentenceTransformer
        import numpy as np
        
        self.questions = questions
        self.answers = answers
        self.model = SentenceTransformer(model_name)
        self.question_embeddings = self.model.encode(questions, convert_to_tensor=True)
        
    def __call__(self, query, top_k=1):
        """Query the retriever."""
        import numpy as np
        
        # Encode the query
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        
        # Compute similarities
        similarities = []
        for i, question_embedding in enumerate(self.question_embeddings):
            similarity = np.dot(query_embedding, question_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(question_embedding)
            )
            similarities.append((self.questions[i], self.answers[i], float(similarity)))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[2], reverse=True)
        
        # Return top_k results
        if top_k == 1:
            return similarities[0] if similarities else (None, None, 0.0)
        else:
            return similarities[:top_k]


class TfidfRetriever:
    """Class-based retriever using TF-IDF for similarity."""
    
    def __init__(self, questions, answers):
        """Initialize with questions and answers."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        self.questions = questions
        self.answers = answers
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.question_vectors = self.vectorizer.fit_transform(questions)
        
    def __call__(self, query, top_k=1):
        """Query the retriever."""
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Vectorize the query
        query_vector = self.vectorizer.transform([query])
        
        # Compute similarities
        similarities = cosine_similarity(query_vector, self.question_vectors)[0]
        
        # Create similarity tuples
        similarity_tuples = [(self.questions[i], self.answers[i], float(similarities[i])) 
                            for i in range(len(self.questions))]
        
        # Sort by similarity (highest first)
        similarity_tuples.sort(key=lambda x: x[2], reverse=True)
        
        # Return top_k results
        if top_k == 1:
            return similarity_tuples[0] if similarity_tuples else (None, None, 0.0)
        else:
            return similarity_tuples[:top_k]


class SimpleRetriever:
    """Class-based retriever using simple word overlap for similarity."""
    
    def __init__(self, questions, answers):
        """Initialize with questions and answers."""
        self.questions = questions
        self.answers = answers
        
        # Process the questions to create normalized versions for matching
        self.processed_questions = []
        for question in questions:
            # Convert to lowercase
            processed = question.lower()
            # Remove punctuation
            for punct in '.,;:!?"\'()[]{}':
                processed = processed.replace(punct, ' ')
            # Split into words and keep unique ones
            words = set(processed.split())
            self.processed_questions.append(words)
    
    def __call__(self, query, top_k=1):
        """Query the retriever."""
        # Process query the same way
        processed_query = query.lower()
        for punct in '.,;:!?"\'()[]{}':
            processed_query = processed_query.replace(punct, ' ')
        query_words = set(processed_query.split())
        
        # Compute similarities
        similarities = []
        for i, question in enumerate(self.questions):
            question_words = self.processed_questions[i]
            
            if not query_words or not question_words:
                score = 0.0
            else:
                # Jaccard similarity: intersection over union
                intersection = len(query_words.intersection(question_words))
                union = len(query_words.union(question_words))
                score = intersection / union if union > 0 else 0.0
                
            similarities.append((question, self.answers[i], score))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[2], reverse=True)
        
        # Return top_k results
        if top_k == 1:
            return similarities[0] if similarities else (None, None, 0.0)
        else:
            return similarities[:top_k]


def build_qa_retrieval_from_json(
    json_paths: Union[str, List[str]], 
    method: str = "embedding", 
    embedding_model: str = "all-MiniLM-L6-v2",
    cache_dir: str = "data/cache",
    force_rebuild: bool = False,
    verbose: bool = True
) -> Tuple[Dict[str, str], Any]:
    """
    Load question-answer pairs from JSON files and build a retrieval system with caching.
    
    Args:
        json_paths: Path to a single JSON file or a list of paths
        method: Similarity method - "embedding", "tfidf", or "simple"
        embedding_model: Name of the embedding model (for "embedding" method)
        cache_dir: Directory to store cached embeddings and retrieval systems
        force_rebuild: If True, rebuilds the system even if cache exists
        verbose: If True, prints status messages
    
    Returns:
        A tuple of (qa_dict, retriever_object)
    """
    # Convert single path to list
    if isinstance(json_paths, str):
        json_paths = [json_paths]
    
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate a unique cache key based on files and method
    cache_key = _generate_cache_key(json_paths, method, embedding_model)
    qa_cache_path = os.path.join(cache_dir, f"{cache_key}_qa.pkl")
    system_cache_path = os.path.join(cache_dir, f"{cache_key}_system.pkl")
    
    # Try to load from cache if not forced to rebuild
    if not force_rebuild and os.path.exists(qa_cache_path) and os.path.exists(system_cache_path):
        if verbose:
            log_event(f"Loading cached QA pairs and retrieval system...")
        
        try:
            with open(qa_cache_path, 'rb') as f:
                qa_dict = pickle.load(f)
            
            with open(system_cache_path, 'rb') as f:
                retriever = pickle.load(f)
            
            if verbose:
                log_event(f"Successfully loaded cached retrieval system with {len(qa_dict)} QA pairs")
            
            return qa_dict, retriever
        except Exception as e:
            if verbose:
                log_event(f"Error loading from cache: {str(e)}")
                log_event("Rebuilding retrieval system...")
    
    # Load QA pairs from JSON files
    qa_dict = {}
    for path in json_paths:
        if verbose:
            log_event(f"Loading QA pairs from {path}...")
        new_pairs = load_qa_from_json(path, verbose=verbose)
        qa_dict.update(new_pairs)
    
    if verbose:
        log_event(f"Loaded {len(qa_dict)} total QA pairs")
    
    # Extract questions and answers
    questions = list(qa_dict.keys())
    answers = list(qa_dict.values())
    
    # Build the retrieval system based on the chosen method
    try:
        if method == "embedding":
            if verbose:
                log_event(f"Building embedding-based retrieval system with model: {embedding_model}")
            retriever = EmbeddingRetriever(questions, answers, embedding_model)
        elif method == "tfidf":
            if verbose:
                log_event("Building TF-IDF based retrieval system")
            retriever = TfidfRetriever(questions, answers)
        else:
            if verbose:
                log_event("Building simple word-overlap based retrieval system")
            retriever = SimpleRetriever(questions, answers)
    except Exception as e:
        if verbose:
            log_event(f"Error building {method} retriever: {str(e)}")
            log_event("Falling back to simple retriever...")
        retriever = SimpleRetriever(questions, answers)
    
    # Cache the QA dict and retrieval system
    try:
        with open(qa_cache_path, 'wb') as f:
            pickle.dump(qa_dict, f)
        
        with open(system_cache_path, 'wb') as f:
            pickle.dump(retriever, f)
        
        if verbose:
            log_event(f"Cached retrieval system to {cache_dir}")
    except Exception as e:
        if verbose:
            log_event(f"Error caching retrieval system: {str(e)}")
    
    return qa_dict, retriever

def load_qa_from_json(json_path: str, verbose: bool = True) -> Dict[str, str]:
    """
    Loads a JSON file with a nested structure of categories containing question-answer pairs
    and converts it to a flat dictionary of questions and answers.
    
    Args:
        json_path: Path to the JSON file
        verbose: If True, prints status messages
    
    Returns:
        A flat dictionary with questions as keys and answers as values
    """
    try:
        # Load the JSON file
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Initialize an empty dictionary for the flat structure
        qa_dict = {}
        
        # Iterate through each category and extract question-answer pairs
        for category, qa_pairs in data.items():
            for question, answer in qa_pairs.items():
                qa_dict[question] = answer
        
        if verbose:
            log_event(f"Loaded {len(qa_dict)} question-answer pairs from {json_path}")
        
        return qa_dict
    
    except Exception as e:
        if verbose:
            log_event(f"Error loading JSON file: {str(e)}")
        return {}

def _generate_cache_key(json_paths: List[str], method: str, embedding_model: str) -> str:
    """Generate a unique cache key based on input parameters."""
    # Get modification timestamps for all JSON files
    timestamps = []
    for path in json_paths:
        if os.path.exists(path):
            timestamps.append(str(os.path.getmtime(path)))
    
    # Create a string that encodes all relevant parameters
    key_str = f"{'-'.join(sorted(json_paths))}-{method}-{embedding_model}-{'-'.join(timestamps)}"
    
    # Hash it to get a fixed-length identifier
    return f"{method}_{embedding_model.replace('/', '_')}_{hashlib.md5(key_str.encode()).hexdigest()[:10]}"

# Assume this function is in nanobioagent/tools/gene_utils.py
def _resolve_path_internal(path: str, project_root=None):
    """
    Internal function to resolve a single path to absolute path.
    Args:
        path: Path to resolve (must be a string)
        project_root: Project root directory (auto-detected if None)
    Returns:
        str: Absolute path
    """
    from pathlib import Path
    import os
    
    # Auto-detect project root if not provided
    if project_root is None:
        # Assume this function is in nanobioagent/tools/gene_utils.py
        project_root = Path(__file__).parent.parent.parent
    else:
        project_root = Path(project_root)
    
    # If already absolute, return as-is
    if os.path.isabs(path):
        return path
    
    # Convert relative path to absolute
    return str(project_root / path)


def resolve_single_path(path: Optional[str], default: Optional[str] = None, project_root=None) -> str:
    # If path is None, use default
    actual_path = path if path is not None else default
    if actual_path is None:
        raise ValueError("Path cannot be None when no default is provided")
    return _resolve_path_internal(actual_path, project_root)


def resolve_multiple_paths(paths: Optional[List[str]], defaults: Optional[List[str]] = None, project_root=None) -> List[str]:
    # If paths is None, use defaults
    actual_paths = paths if paths is not None else defaults
    if actual_paths is None:
        raise ValueError("Paths cannot be None when no defaults are provided")
    return [_resolve_path_internal(p, project_root) for p in actual_paths]