SYSTEM_PROMPT_NCBI = 'You are a helpful assistant.\nWhen you have completed your answer, write "Answer: [your answer]" followed by two newlines (i.e. "\n\n") to indicate completion.\nFor accuracy, you should use the NCBI Web APIs, Eutils and BLAST (with examples in user prompt), to obtain the data.\n'
SYSTEM_PROMPT_BASIC = 'You are a helpful assistant.'
STOP_SEQUENCE=['->', '\n\nQuestion']
SECONDS_BETWEEN_CALLS = 1.1 # 3.1
# Token estimation constants - centralized configuration calibrated from empirical tests on NCBI documents
TOKEN_ESTIMATION_CONFIG = {
    "default_chars_per_token": 3.0,  # Conservative average across models
    "model_specific_ratios": {
        "mistralai": 2.18,
        "tiiuae": 2.32,
        "ibm": 2.33,
        "google/gemma-2": 1.80,
        "google": 2.40,
        "qwen": 2.49,
        "gpt": 2.99,      # GPT models are more efficient
        "microsoft": 2.98,
        "meta": 3.00,
        "nvidia/llama": 3.00,
        "nvidia": 2.41,
        "claude": 3.00      # 4.61 but let's be conservative
    },
    "minimum_tokens": 1
}

import json
import os
import time
import re
from pathlib import Path
from typing import Dict, Any, Union, List, Optional
from datetime import datetime
from dotenv import load_dotenv
from ..tools.gene_utils import log_event

load_dotenv()

import torch
from langchain_community.llms import OpenAI
from langchain_huggingface import HuggingFacePipeline
from langchain_community.chat_models import ChatOpenAI, ChatGooglePalm, ChatHuggingFace
from langchain_ollama import ChatOllama 
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
LANGCHAIN_AVAILABLE = True

# To-Do: fix the deprecated warning from ChatOpenAI
# LangChainDeprecationWarning: The class `ChatOpenAI` was deprecated in LangChain 0.0.10 and will be removed in 1.0. An updated version of the class exists in the :class:`~langchain-openai package and should be used instead.
import warnings
warnings.filterwarnings("ignore", message="Convert_system_message_to_human will be deprecated!")
warnings.filterwarnings("ignore", message=".*ChatOpenAI.*deprecated.*", category=DeprecationWarning)

# Returns model-specific configurations from JSON file, includes context windows, pricing data, and dynamic cut_length calculation
def get_model_config(model_name, config_file="model_config.json"):
    # Use consistent path resolution - always look in config directory
    config_dir = Path(__file__).parent.parent.joinpath("config")
    config_path = config_dir.joinpath(config_file)
    # Load config file
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
    except FileNotFoundError:
        log_event(f"Config file {config_path} not found. Using inline defaults.")
        config_data = {}
    except Exception as e:
        log_event(f"Error loading config from {config_path}: {e}. Using inline defaults.")
        config_data = {}
    
    # Get defaults and nvidia prefixes
    defaults = config_data.get("_defaults", {
        "cut_length": 18000,
        "max_tokens": 512,
        "model_type": "chat",
        "temperature": 0,
        "context_window": 32000,
        "pricing": {"input": 0.001, "output": 0.002, "per_tokens": 1000, "currency": "USD"}
    })
    
    nvidia_nim_prefixes = config_data.get("nvidia_nim_prefixes", [])
    models = config_data.get("models", {})
    config = defaults.copy()
    
    if model_name.startswith("nvidia_nim/"):
        config["model_type"] = "nvidia_nim"
    elif any(model_name.startswith(prefix) for prefix in nvidia_nim_prefixes):
        config["model_type"] = "nvidia_nim"
    
    if model_name.startswith("huggingface/"):
        config["model_type"] = "huggingface"
    
    # potentially overwrite for specific behaviour
    if model_name in models:
        config.update(models[model_name])
    else:
        # Try prefix matching (maintains existing behavior). Consider exact match later?
        for model_key in models.keys():
            if model_name.startswith(model_key):
                config.update(models[model_key])
                break
        else:
            # No match found
            log_event(f"No config found for model {model_name}, using defaults")
    
    # Dynamic cut_length calculation: context_window - max_tokens - buffer
    # Note: Convert from tokens to characters since cut_length is used for character truncation
    if "context_window" in config and "max_tokens" in config:
        buffer = 1000  # Safety buffer in tokens
        calculated_cut_length_tokens = config["context_window"] - config["max_tokens"] - buffer
        
        # Convert tokens to characters using centralized conversion function
        calculated_cut_length_chars = tokens_to_chars(calculated_cut_length_tokens, model_name)
        
        # Use the smaller of configured cut_length or calculated value
        if "cut_length" not in config or config["cut_length"] > calculated_cut_length_chars:
            # Minimum based on model-specific ratio
            min_chars = tokens_to_chars(1000, model_name)  # 1000 tokens minimum
            config["cut_length"] = max(calculated_cut_length_chars, min_chars)
    
    # Post-processing with metadata
    config["_metadata"] = {
        "matched_key": model_name if model_name in models else "prefix_match",
        "auto_detected_nvidia": any(model_name.startswith(prefix) for prefix in nvidia_nim_prefixes),
        "calculated_cut_length": config.get("cut_length")
    }
    
    return config

# Create a LangChain model based on the model name and configuration.
def create_langchain_model(model_name, config_data=None):
    if not LANGCHAIN_AVAILABLE:
        return None
        
    model_config = get_model_config(model_name)
    model_type = model_config.get("model_type", "chat")
    temperature = model_config.get("temperature", 0)
    
    # Get API keys from environment variables or .env
    api_keys = {
        'openai': os.environ.get('OPENAI_API_KEY'),
        'anthropic': os.environ.get('ANTHROPIC_API_KEY'),
        'nvidia': os.environ.get('NVIDIA_API_KEY'),
        'google': os.environ.get('GOOGLE_API_KEY'),
        'huggingface': os.environ.get('HUGGINGFACE_API_KEY')
    }
    
    if model_type == "nvidia_nim":
        try:
            nvidia_api_key = api_keys.get('nvidia', os.environ.get('NVIDIA_API_KEY'))
            if not nvidia_api_key:
                log_event("No NVIDIA API key found. Please provide one in the config file or set the NVIDIA_API_KEY environment variable.")
                return None
            
            use_own_wrapper_for_max_tokens = True
            max_tokens= model_config.get("max_tokens", 256)
            '''
            if "8b" in model_name.lower() or "nano" in model_name.lower():
                use_own_wrapper_for_max_tokens = True
                max_tokens = 512 # 256
            if model_name.startswith("llama-3"): # legacy convention, to remove later
                model_identifier = "nvidia/" + model_name  # Add nvidia/ prefix for Llama models
            else:
                model_identifier = model_name  # Use as-is: "ibm/granite-3.3-8b-instruct"
            '''
            model_identifier = model_name
            # Create ChatOpenAI with NVIDIA NIM base URL
            llm = ChatOpenAI(
                model= model_identifier,
                temperature=temperature,
                openai_api_key=nvidia_api_key,
                openai_api_base="https://integrate.api.nvidia.com/v1"
            )
            if use_own_wrapper_for_max_tokens:
                wrapped_llm = _wrap_nvidia_model(llm, max_tokens=max_tokens)
                return wrapped_llm
            else:
                return llm
            
        except Exception as e:
            log_event(f"Error creating NVIDIA NIM model: {e}")
            return None

    # OpenAI Chat models
    elif model_type == "openai":
        openai_api_key = api_keys.get('openai') or os.environ.get('OPENAI_API_KEY')
        if model_name.startswith("gpt-5"):
            return ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                model_kwargs={"max_completion_tokens": model_config["max_tokens"]},
                # max_completion_tokens=model_config["max_tokens"],
                openai_api_key=openai_api_key
            )
        else:
            return ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                max_tokens=model_config["max_tokens"],
                openai_api_key=openai_api_key
            )
	
	# Anthropic Claude models
    elif model_type == "anthropic":
        anthropic_api_key = api_keys.get('anthropic', os.environ.get('ANTHROPIC_API_KEY'))
        if not anthropic_api_key:
            log_event("No Anthropic API key found. Please set the ANTHROPIC_API_KEY environment variable.")
            return None
		
        return ChatAnthropic(
			model_name=model_name,
			temperature=temperature,
            timeout=180,
            stop=None,
			max_tokens=model_config["max_tokens"],
			anthropic_api_key=anthropic_api_key
		)
    
    # Google Gemini models 
    elif model_type == "google":
        google_api_key = api_keys.get('google', os.environ.get('GOOGLE_API_KEY'))
        if not google_api_key:
            log_event("No Google API key found. Please set the GOOGLE_API_KEY environment variable.")
            return None
        
        # Instantiate the ChatGoogleGenerativeAI class
        return ChatGoogleGenerativeAI(
            model=model_name,  # e.g., "gemini-1.5-flash-latest"
            google_api_key=google_api_key,
            temperature=temperature,
            max_output_tokens=model_config.get("max_tokens"),
            convert_system_message_to_human=True # Recommended for compatibility
        )
    
    # Hugging Face Hub models (Chat Mode)
    elif model_type == "huggingface":
        huggingface_api_key = api_keys.get('huggingface', os.environ.get('HUGGINGFACE_API_KEY'))
        if not huggingface_api_key:
            log_event("No Hugging Face API key found. Please set the HUGGINGFACE_API_KEY environment variable.")
            return None
        # remove the prefix 
        if model_name.startswith("huggingface/"):
            model_name = model_name.replace("huggingface/", "")    
        
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=huggingface_api_key,
            openai_api_base="https://router.huggingface.co/v1",
            max_tokens=model_config.get("max_tokens")
        )

    
    # In the create_langchain_model function, modify the Ollama section:
    elif "ollama" in model_name.lower():
        try:
            # Extract the actual model name if prefixed
            if "/" in model_name:
                _, ollama_model = model_name.split("/")
            else:
                ollama_model = model_name
                
            log_event(f"Loading Ollama model: {ollama_model}")
            # Return ChatOllama instead of Ollama
            return ChatOllama(
                model=ollama_model,
                temperature=temperature,
                num_predict=model_config["max_tokens"]
            )
        except Exception as e:
            log_event(f"Error loading Ollama model: {e}")
            return None
    
    elif model_type == "local":
        # For models like "meditron-7b", extract the path or use the name directly
        model_path = model_name
        if "/" not in model_name:
            # Re-testing needed, as I did lots of refactoring but have no time or need to re-test the local mode
            if model_name.startswith("meditron"):
                model_path = f"epfl-llm/{model_name}"
            elif model_name.startswith("llama"):
                model_path = f"meta-llama/{model_name}"
            elif model_name.startswith("mistral"):
                model_path = f"mistralai/{model_name}"
                
        return load_huggingface_model(model_path)
    
    # OpenAI Completion models (older models)
    elif model_type == "completion":
        openai_api_key = api_keys.get('openai') or os.environ.get('OPENAI_API_KEY')
        return OpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=model_config["max_tokens"],
            openai_api_key=openai_api_key
        )
    
    else:
        log_event(f"Model type {model_type} not supported for {model_name}")
        return None

# Call a LangChain model with the given prompt with the hard-coded rate-limiting logic similar to genegpt approach
def call_langchain_model(llm, q_prompt, system_message=SYSTEM_PROMPT_NCBI, model_name=None):
    # Add rate limiting
    if 'prev_call_time' in globals():
        delta = time.time() - globals()['prev_call_time']
        if delta < SECONDS_BETWEEN_CALLS:  # Ensure at least 3.1 seconds between calls
            time.sleep(SECONDS_BETWEEN_CALLS- delta)
    globals()['prev_call_time'] = time.time()
    
    if llm is None:
        return "Error: No LangChain model available."
    
    # Thinking control = True means letting the model config drive the behaviour. False means we don't interfere and let the model do its thing internally
    enable_thinking_control = True
    # To-Do: Disable system messages for Gemma models that don't support them, move to config later
    models_no_system = ["google/gemma-2-2b-it", "google/gemma-7b", "google/codegemma-7b", "google/gemma-2-9b-it", "gpt-3.5-turbo"]
    if model_name in models_no_system:
        system_message = ''

    try:
        # For chat models
        if isinstance(llm, (ChatOpenAI, ChatAnthropic, ChatOllama, ChatGoogleGenerativeAI, ChatGooglePalm, ChatHuggingFace)):
            if model_name:
                model_config = get_model_config(model_name)
                if enable_thinking_control:
                    thinking_control_prompt = model_config.get("thinking_control_prompt")
                    log_event(f"call_langchain_model: model_name: {model_name} + thinking_control_prompt: {thinking_control_prompt}")
                    if thinking_control_prompt:
                        system_message = system_message + " " + thinking_control_prompt

            if system_message and system_message.strip():
                messages = [
                    SystemMessage(content=system_message),
                    HumanMessage(content=q_prompt)
                ]
            else:
                messages = [
                    HumanMessage(content=q_prompt)
                ]
            
            response = llm.invoke(messages, stop=STOP_SEQUENCE)
            return response.content
            # remove the think tag
            # return re.sub(r'<think>.*?</think>', '', response.content, flags=re.DOTALL)
        
        # For completion models
        else:
            prompt_template = PromptTemplate(
                input_variables=["prompt"],
                template="{prompt}"
            )
            chain = prompt_template | llm.bind(stop=STOP_SEQUENCE) | StrOutputParser()
            return chain.invoke({"prompt": q_prompt})
            
    except Exception as e:
        log_event(f"Error calling LangChain model: {e}")
        return f"Error: {e}"

# used by geneGPT method to call LLM
def call_llm(q_prompt, model_name, config_data=None, use_fallback=False, cost_tracker=None):
    """
    Call the selected language model with the given prompt.
    """
    model_config = get_model_config(model_name)
    # log_event(f"Model config of {model_name} model: \n{model_config}")
    # Truncate if necessary
    if len(q_prompt) > model_config["cut_length"]:
        # truncate from the start
        q_prompt = q_prompt[len(q_prompt) - model_config["cut_length"]:]
    # use_fallback is only for gpt-x models for testing purpose, so force it to False for others to always invoke via langchain 
    if not model_name.startswith("gpt-"):
        use_fallback = False

    try:
        if LANGCHAIN_AVAILABLE and not use_fallback:
            # Create or get the LangChain model
            # log_event(f"Creating model via LangChain: {model_name}")
            log_event(f"call_llm() q_prompt: {q_prompt}")
            log_event(f"call_llm() model_name: {model_name}")
            
            model = create_langchain_model(model_name, config_data)
            if model:
                response = call_langchain_model(model, q_prompt, model_name=model_name)
                # Track costs if tracker provided
                if cost_tracker:
                    cost_tracker.log_call(model_name, q_prompt, response)
                
                log_event(f"call_llm() response: {response}")
                return response
            else:
                raise Exception(f"Failed to create LangChain model for {model_name}")
        else:
            # Fallback to direct OpenAI call as per geneGPT original method
            if not model_name.startswith("gpt-"):
                # log_event(f"Warning: LangChain not available. Falling back to gpt-3.5-turbo.")
                # model_name = "gpt-3.5-turbo"
                raise Exception(f"fallback only used for testing purposes for gpt-x models, not for {model_name}")
            
            # Handle rate limiting    
            prev_call_time = globals().get('prev_call_time', 0)
            delta = time.time() - prev_call_time
            if delta < SECONDS_BETWEEN_CALLS:
                time.sleep(SECONDS_BETWEEN_CALLS - delta)

            globals()['prev_call_time'] = time.time()
            response = None
            try:
                prev_call = time.time()
                
                # Check OpenAI version
                import openai
                openai_version = openai.__version__
                
                if openai_version.startswith("0."):
                    log_event("using old API: OpenAI SDK (v0.x)")
                    response = openai.ChatCompletion.create(
                        model=model_name,
                        messages=[
                            # {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": q_prompt}
                        ],
                        max_tokens=model_config["max_tokens"],
                        temperature=0,
                        stop=STOP_SEQUENCE,
                        n=1
                    )
                    response_text = response['choices'][0]['message']['content']
                else:
                    log_event("using new API:OpenAI SDK (v1.x)")
                    #client = openai.OpenAI(api_key=local_config.API_KEY)
                    openai_api_key = os.environ.get('OPENAI_API_KEY')
                    client = openai.OpenAI(api_key=openai_api_key)
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "user", "content": q_prompt}
                        ],
                        max_tokens=model_config["max_tokens"],
                        temperature=0,
                        stop=STOP_SEQUENCE
                    )
                    response_text = response.choices[0].message.content
                
                # Track costs if tracker provided
                if cost_tracker:
                    # For OpenAI direct calls, try to get actual usage
                    actual_usage = None
                    if hasattr(response, 'usage'):
                        actual_usage = {
                            "input_tokens": response.usage.prompt_tokens,
                            "output_tokens": response.usage.completion_tokens
                        }
                    cost_tracker.log_call(model_name, q_prompt, response_text, actual_usage)
                
                return response_text
                    
            except Exception as e:
                log_event(f"Error in OpenAI fallback: {e}")
                return f"Error: {e}"
    except Exception as e:
        log_event(f"Error calling model {model_name}: {e}")
        return f"Error: {e}"

def invoke_llm_with_config(llm, prompt_text, model_name, system_message = SYSTEM_PROMPT_BASIC) -> Any:
    """
    Unified LLM invocation with all config applied consistently.
    Used by both agent framework and tools.
    """
    model_config = get_model_config(model_name)
    
    # Apply thinking control
    effective_system_message = system_message or ""
    thinking_control_prompt = model_config.get("thinking_control_prompt")
    if thinking_control_prompt:
        effective_system_message += " " + thinking_control_prompt
        log_event(f"SystemMessage: {effective_system_message}")
    
    # Apply truncation
    #if len(prompt_text) > model_config["cut_length"]:
    #    prompt_text = prompt_text[len(prompt_text) - model_config["cut_length"]:]
    
    # To-Do: Disable system messages for Gemma models that don't support them, move to config later
    models_no_system = ["google/gemma-2-2b-it", "google/gemma-7b", "google/codegemma-7b", "google/gemma-2-9b-it"]
    if model_name in models_no_system:
        effective_system_message = None
        
    # Create messages
    if effective_system_message:
        messages = [
            SystemMessage(content=effective_system_message),
            HumanMessage(content=prompt_text)
        ]
        response = llm.invoke(messages)
        log_event(f"invoke_llm_with_config: with system message: {effective_system_message}")
    else:
        response = llm.invoke(prompt_text)
        log_event(f"invoke_llm_with_config: with prompt")

    return clean_llm_response(response.content)

# Remove thinking tags and extract clean answer.
def clean_llm_response(response: str) -> str:
    # Remove <think>...</think> blocks (including empty ones)
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    
    # Clean up extra whitespace and newlines
    response = response.strip()
    
    return response

# Utility functions for estimation of costs
def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> Dict[str, Any]:
    """Calculate cost for a model based on token usage."""
    config = get_model_config(model_name)
    pricing = config.get("pricing", {})
    
    if not pricing:
        return {"total_cost": 0.0, "currency": "USD", "breakdown": {}}
    
    per_tokens = pricing.get("per_tokens", 1000)
    input_rate = pricing.get("input", 0.0)
    output_rate = pricing.get("output", 0.0)
    currency = pricing.get("currency", "USD")
    
    input_cost = (input_tokens / per_tokens) * input_rate
    output_cost = (output_tokens / per_tokens) * output_rate
    total_cost = input_cost + output_cost
    
    return {
        "total_cost": round(total_cost, 6),
        "currency": currency,
        "breakdown": {
            "input_cost": round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "rates": {"input": input_rate, "output": output_rate, "per_tokens": per_tokens}
        }
    }

def get_chars_per_token(model_name: str = "") -> float:
    """
    Get the characters per token ratio for a specific model.
    Centralized function to ensure consistency across all token calculations.
    """
    if not model_name:
        return TOKEN_ESTIMATION_CONFIG["default_chars_per_token"]
    # model_lower = model_name.lower()
    model_lower = model_name.lower().replace("huggingface/", "")
    ratios = TOKEN_ESTIMATION_CONFIG["model_specific_ratios"]
    # Check for model prefixes
    for model_type, ratio in ratios.items():
        if model_type in model_lower:
            return ratio
    return TOKEN_ESTIMATION_CONFIG["default_chars_per_token"]

def estimate_tokens(text: str, model_name: str = "") -> int:
    if not text:
        return 0
    chars_per_token = get_chars_per_token(model_name)
    estimated_tokens = len(text) / chars_per_token
    return max(TOKEN_ESTIMATION_CONFIG["minimum_tokens"], int(estimated_tokens))

def tokens_to_chars(tokens: int, model_name: str = "") -> int:
    if tokens <= 0:
        return 0
    chars_per_token = get_chars_per_token(model_name)
    return int(tokens * chars_per_token)

def chars_to_tokens(chars: int, model_name: str = "") -> int:
    if chars <= 0:
        return 0
    
    chars_per_token = get_chars_per_token(model_name)
    estimated_tokens = chars / chars_per_token
    return max(TOKEN_ESTIMATION_CONFIG["minimum_tokens"], int(estimated_tokens))

def get_context_utilization(model_name: str, current_tokens: int) -> Dict[str, Any]:
    """Get context window utilization for a model."""
    config = get_model_config(model_name)
    context_window = config.get("context_window", 32000)
    
    utilization_percent = (current_tokens / context_window) * 100
    
    return {
        "current_tokens": current_tokens,
        "context_window": context_window,
        "utilization_percent": round(utilization_percent, 2),
        "remaining_tokens": context_window - current_tokens,
        "status": "high" if utilization_percent > 80 else "medium" if utilization_percent > 50 else "low"
    }

# UniversalCostTracker Class for Phase 2
class UniversalCostTracker:
    """Track token usage and costs across all LLM interactions."""
    
    def __init__(self):
        self.calls = []
        self.session_start = datetime.now()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.peak_context_usage = 0.0
        self.models_used = set()
    
    def log_call(self, model_name: str, input_text: str, output_text: str, actual_usage: Optional[Dict[str, int]] = None):
        """Log an LLM call with cost and usage tracking."""
        timestamp = datetime.now()
        
        # Use actual usage if provided, otherwise estimate using centralized function
        if actual_usage:
            input_tokens = actual_usage.get("input_tokens", 0)
            output_tokens = actual_usage.get("output_tokens", 0)
        else:
            input_tokens = estimate_tokens(input_text, model_name)
            output_tokens = estimate_tokens(output_text, model_name)
        
        # Calculate cost
        cost_info = calculate_cost(model_name, input_tokens, output_tokens)
        
        # Calculate context utilization
        total_tokens = input_tokens + output_tokens
        context_info = get_context_utilization(model_name, total_tokens)
        
        # Store call information
        call_data = {
            "timestamp": timestamp,
            "model_name": model_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cost": cost_info["total_cost"],
            "currency": cost_info["currency"],
            "context_utilization": context_info["utilization_percent"],
            "cost_breakdown": cost_info["breakdown"]
        }
        
        self.calls.append(call_data)
        
        # Update session totals
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += cost_info["total_cost"]
        self.peak_context_usage = max(self.peak_context_usage, context_info["utilization_percent"])
        self.models_used.add(model_name)
    
    def get_usage_summary(self) -> str:
        """Get formatted usage summary for console/log output."""
        if not self.calls:
            return "No LLM calls tracked."
        
        total_tokens = self.total_input_tokens + self.total_output_tokens
        call_count = len(self.calls)
        
        summary = []
        summary.append("=== Token Usage Summary ===")
        summary.append(f"Total Calls: {call_count}")
        summary.append(f"Total Tokens: {total_tokens:,} ({self.total_input_tokens:,} input + {self.total_output_tokens:,} output)")
        summary.append(f"Total Cost: ${self.total_cost:.9f}")
        summary.append(f"Peak Context Usage: {self.peak_context_usage:.2f}%")
        summary.append(f"Models Used: {', '.join(sorted(self.models_used))}")
        
        # Call breakdown
        summary.append("\n=== Call Breakdown ===")
        for i, call in enumerate(self.calls, 1):
            summary.append(f"Call #{i} ({call['model_name']}): {call['total_tokens']:,} tokens, "
                        f"${call['cost']:.9f}, {call['context_utilization']:.2f}% context")
        
        # Optimization suggestions
        optimization = self.get_optimization_suggestions()
        if optimization:
            summary.append("\n=== Cost Optimization ===")
            summary.extend(optimization)
        
        return "\n".join(summary)
    
    def get_output_summary(self) -> List[str]:
        """Get structured list for function returns."""
        if not self.calls:
            return ["No LLM calls tracked."]
        
        total_tokens = self.total_input_tokens + self.total_output_tokens
        call_count = len(self.calls)
        
        summary = [
            f"=== Token Usage ===",
            f"Total Calls: {call_count}",
            f"Total Tokens: {total_tokens:,} ({self.total_input_tokens:,} input + {self.total_output_tokens:,} output)",
            f"Total Cost: ${self.total_cost:.9f}",
            f"Peak Context Usage: {self.peak_context_usage:.2f}%"
        ]
        
        # Add call breakdown
        summary.append("=== Call Breakdown ===")
        for i, call in enumerate(self.calls, 1):
            summary.append(f"Call #{i} ({call['model_name']}): {call['total_tokens']:,} tokens, "
                        f"${call['cost']:.9f}, {call['context_utilization']:.2f}% context")
        
        # Add optimization suggestions
        optimization = self.get_optimization_suggestions()
        if optimization:
            summary.append("=== Cost Optimization ===")
            summary.extend(optimization)
        
        return summary
    
    def get_optimization_suggestions(self) -> List[str]:
        """Generate cost reduction recommendations."""
        if not self.calls or len(self.calls) < 2:
            return []
        
        suggestions = []
        
        # Analyze model efficiency
        model_stats = {}
        for call in self.calls:
            model = call["model_name"]
            if model not in model_stats:
                model_stats[model] = {"calls": 0, "total_cost": 0.0, "total_tokens": 0}
            model_stats[model]["calls"] += 1
            model_stats[model]["total_cost"] += call["cost"]
            model_stats[model]["total_tokens"] += call["total_tokens"]
        
        # Calculate cost per token for each model
        model_efficiency = {}
        for model, stats in model_stats.items():
            if stats["total_tokens"] > 0:
                model_efficiency[model] = stats["total_cost"] / stats["total_tokens"]
        
        # Find cheaper alternatives
        if len(model_efficiency) > 1:
            sorted_models = sorted(model_efficiency.items(), key=lambda x: x[1])
            cheapest = sorted_models[0][0]
            most_expensive = sorted_models[-1][0]
            
            cost_ratio = model_efficiency[most_expensive] / model_efficiency[cheapest]
            if cost_ratio > 2:
                suggestions.append(f"🔄 Consider using {cheapest} for simple queries ({cost_ratio:.1f}x cheaper than {most_expensive})")
        
        # Context usage warnings
        high_usage_calls = [call for call in self.calls if call["context_utilization"] > 80]
        if high_usage_calls:
            suggestions.append(f"⚠️  {len(high_usage_calls)} calls used >80% context window - consider input truncation")
        
        # Cost alerts
        if self.total_cost > 1.0:
            suggestions.append(f"💰 Session cost ${self.total_cost:.2f} - monitor usage for production workloads")
        
        return suggestions

def get_hf_task_for_model(model_name):
    """
    Determine the appropriate HuggingFace task for a given model.
    """
    model_lower = model_name.lower()
    
    # Models that typically use "conversational" task
    conversational_indicators = [
        "nemotron", "chat", "instruct", "assistant", "conversational",
        "dialogue", "conv", "-it", "-chat-hf", "-instruct-hf"
    ]
    
    for indicator in conversational_indicators:
        if indicator in model_lower:
            log_event(f"Detected conversational model due to '{indicator}' in {model_name}")
            return "conversational"
    
    # Default to text-generation for most models
    log_event(f"Using text-generation task for {model_name}")
    return "text-generation"

def load_huggingface_model(model_path):
    """
    Load a HuggingFace model for local inference.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        
        # Load the model and tokenizer
        log_event(f"Loading model from {model_path}. This might take a while...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Create a pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.0,
            top_p=0.95,
            repetition_penalty=1.15
        )
        
        # Create a HF pipeline LLM
        llm = HuggingFacePipeline(pipeline=pipe)
        return llm
    except Exception as e:
        log_event(f"Error loading HuggingFace model: {e}")
        return None
    
# wraps existing ChatOpenAI model to properly handle params like max_tokens for NVIDIA API
def _wrap_nvidia_model(llm, max_tokens=256):
    import functools
    try:
        # Store original create method directly from the completions object
        original_create = llm.client.create
        @functools.wraps(original_create)
        def patched_create(*args, **kwargs):
            # Add max_tokens to the API call
            kwargs['max_tokens'] = max_tokens
            # Remove any parameters that NVIDIA API doesn't accept
            kwargs.pop('max_completion_tokens', None)
            kwargs.pop('n', None)
            kwargs.pop('logprobs', None)
            kwargs.pop('top_logprobs', None)
            return original_create(*args, **kwargs)
        
        # Replace the create method directly
        llm.client.create = patched_create
        
        return llm
        
    except Exception as e:
        log_event(f"Warning: Could not apply NVIDIA wrapper: {e}")
        return llm  # Return unwrapped model if patching fails

# Extract JSON string from LLM response that may contain additional text. Handles common patterns like markdown code blocks and explanatory text.
def extract_json_string_from_llm_response(response_text: str) -> str:
    import re
    import json
    response_text = response_text.strip()
    # Remove markdown code blocks
    response_text = re.sub(r'```(?:json)?\s*', '', response_text)
    response_text = re.sub(r'```\s*', '', response_text)
    
    # Find the first { and last } to extract JSON
    start_idx = response_text.find('{')
    end_idx = response_text.rfind('}')
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        json_str = response_text[start_idx:end_idx + 1].strip()
        try:
            # Validate it's proper JSON
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError:
            pass
    
    # Try with improved regex pattern that handles multiline
    json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(0).strip()
        try:
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError:
            pass
    
    # Last resort: try the whole response
    try:
        json.loads(response_text)
        return response_text
    except json.JSONDecodeError:
        pass
    
    raise ValueError(f"Could not extract valid JSON from LLM response. Response: {response_text[:500]}...")

# Extract and parse JSON from LLM response as a dictionary
def extract_json_dict_from_llm_response(response_text: str, fallback_parser=None, required_fields=None) -> Dict[str, Any]:
    try:
        # existing JSON extraction logic
        json_string = extract_json_string_from_llm_response(response_text)
        json_data = json.loads(json_string)
        
        # Validate that the JSON has required fields
        if required_fields:
            missing_fields = [field for field in required_fields if field not in json_data]
            if missing_fields:
                raise ValueError(f"JSON missing required fields: {missing_fields}")
        
        return json_data
        
    except ValueError as original_error:  # Changed variable name
        # Use the provided fallback parser if available
        if fallback_parser:
            try:
                log_event(f"Calling fallback parser: {fallback_parser.__name__}")
                result = fallback_parser(response_text)
                log_event(f"Fallback parser succeeded: {result}")
                return result
            except Exception as fallback_error:  # Changed variable name
                log_event(f"Fallback parser failed: {fallback_error}")
                pass
        
        # If all else fails, re-raise the original error
        raise original_error