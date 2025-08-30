"""
NanoBioAgent (NBA): Nano-Scale Language Model Agents for Genomics
Built on top of GeneGPT with enhanced multi-agent framework
"""
__version__ = "1.0.0"
__author__ = "George Hong"
# Import core components
from .core.agent_framework import AgentFramework
from .tools.gene_utils import log_event
# Import main API functions
try:
    from .api import gene_answer, gene_compare
    from .api import gene_answer_direct, gene_answer_agent, gene_answer_genegpt, gene_answer_code, geneGPT_answer_retrieve
    API_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import API functions: {e}")
    # Create dummy functions
    def gene_answer(*args, **kwargs):
        return ["", "", "Error: API functions not available", [], [], 0.0]
    def gene_compare(*args, **kwargs):
        return []
    def gene_answer_direct(*args, **kwargs):
        return ["", "", "Error: API functions not available", [], [], 0.0]
    def gene_answer_agent(*args, **kwargs):
        return ["", "", "Error: API functions not available", [], [], 0.0]
    def gene_answer_genegpt(*args, **kwargs):
        return ["", "", "Error: API functions not available", [], [], 0.0]
    def gene_answer_code(*args, **kwargs):
        return ["", "", "Error: API functions not available", [], [], 0.0]
    def geneGPT_answer_retrieve(*args, **kwargs):
        return ["", "", "Error: API functions not available", [], [], 0.0]
    API_AVAILABLE = False

def answer(question: str, model_name: str = "gpt-4o-mini", method: str = "agent", **kwargs) -> str:
    """
    Main API entry point for NanoBioAgent with multiple methods.
    
    Args:
        question: Biological question to answer
        model_name: LLM model to use (default: gpt-4o-mini for nano-scale efficiency)
        method: Method to use - "agent" (default), "genegpt", "direct", "retrieve", "code"
        **kwargs: Additional arguments
        
    Returns:
        str: The answer to the biological question
        
    Examples:
        >>> import nanobioagent as nba
        >>> # Agent method (default - nano-scale intelligence)
        >>> result = nba.answer("What is the official gene symbol of BRCA1?")
        >>> 
        >>> # GeneGPT method (baseline comparison)  
        >>> result = nba.answer("What is BRCA1?", method="genegpt")
        >>>
        >>> # Direct LLM method (no tools)
        >>> result = nba.answer("What is BRCA1?", method="direct")
    """
    try:
        # Use the enhanced gene_answer with specified method
        result = gene_answer(question, method=method, model_name=model_name, **kwargs)
        return result[2] if isinstance(result, list) and len(result) >= 3 else str(result)
    except Exception as e:
        log_event(f"Error in nba.answer(): {e}")
        return f"Error: {str(e)}"

# Make key functions available at package level
__all__ = [
    "answer",           # Simple interface returning answers only  
    "gene_answer",      # Full interface with all results returned
    "gene_compare",     # Compare multiple methods
    "gene_answer_direct", # Direct method
    "gene_answer_agent",  # Agent method
    "gene_answer_genegpt", # GeneGPT method
    "gene_answer_code",   # Code method
    "geneGPT_answer_retrieve", # Retrieve method
    "AgentFramework",
    "log_event"
]