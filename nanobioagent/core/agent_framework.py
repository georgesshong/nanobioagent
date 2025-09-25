"""
Generic Agent Framework using LangChain LCEL patterns.
Completely domain-agnostic and configuration-driven.
"""
DEBUG_PRINT = False # control whether to print debug info for this particular module
DEBUG_PRINT_MAX_CHAR = 2000 # Max chars to print for the function debug_log()
MODE_MONITOR_MEMORY = "Light"  # "None", "Light", "Detailed"
MODE_MONITOR_TIME = "Standard"  # "None", "Standard"
DEFAULT_MODEL_NAME = "gpt-4o-mini" # Default model for LLM calls in this module
FINAL_PARSING_CONFIG = {
    # Field-specific limits
    "field_limits": {
        "document": 6000, # 2000,
        "raw_response": 1000,
        "api_response": 800,
        "logs": 500
    },
    # Total character limit for all execution results
    "max_total_chars": 16000, # 8000,
    # Fields that should never be truncated
    "preserve_fields": ["answer", "parameters", "idlist", "database_used", "success"],
    # Fields to prioritize (keep full if possible)
    "priority_fields": ["answer", "parameters", "document"]
}

import json
import time
import psutil
import os
import re
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from pydantic import BaseModel, Field
from .model_utils import create_langchain_model, extract_json_string_from_llm_response, extract_json_dict_from_llm_response, UniversalCostTracker, invoke_llm_with_config
from ..tools.gene_utils import log_event
from langchain_core.runnables import (
    RunnableSequence, 
    RunnablePassthrough, 
    RunnableLambda
)
from langchain_core.prompts import PromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.tools import Tool
# from langchain_core.runnables.retry import RetryRunnable # new functionality
from langchain_openai import ChatOpenAI

# Debug printing utilities
def debug_log(operation, data, category="GENERAL", show_content=True, max_length=DEBUG_PRINT_MAX_CHAR):
    """Simple debug logging that respects DEBUG_PRINT flag"""
    if not DEBUG_PRINT:
        return
    print(f"🔍 [{category}] {operation}")
    if show_content and data is not None:
        # Smart truncation for long content
        content = str(data)
        if len(content) > max_length:
            content = content[:max_length] + "... [truncated]"
        print(f"   {content}")
    print()
    
# Performance monitoring utilities
def get_memory_usage():
    """Get current process memory usage in MB."""
    if MODE_MONITOR_MEMORY == "None":
        return 0
    
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return round(memory_info.rss / 1024 / 1024, 2)  # MB
    except:
        return 0

# Format timing duration for display in seconds with 3 decimal places.
def format_timing(duration_ms):
    return f"{duration_ms/1000:.3f}s"

class PerformanceMonitor:
    """Lightweight performance monitoring."""
    
    def __init__(self):
        self.metrics = {}
        self.start_memory = get_memory_usage()
    
    def start_timer(self, operation_name: str):
        """Start timing an operation."""
        if MODE_MONITOR_TIME == "None":
            return
        
        self.metrics[operation_name] = {
            "start_time": time.perf_counter(),
            "start_memory": get_memory_usage()
        }
    
    def end_timer(self, operation_name: str):
        """End timing an operation and record metrics."""
        if MODE_MONITOR_TIME == "None" or operation_name not in self.metrics:
            return
        
        end_time = time.perf_counter()
        end_memory = get_memory_usage()
        
        metric = self.metrics[operation_name]
        duration_ms = (end_time - metric["start_time"]) * 1000
        memory_delta = end_memory - metric["start_memory"]
        
        self.metrics[operation_name].update({
            "duration_ms": round(duration_ms, 1),
            "memory_start_mb": metric["start_memory"],
            "memory_end_mb": end_memory,
            "memory_delta_mb": round(memory_delta, 2)
        })
    
    def get_performance_summary(self):
        """Get formatted performance summary."""
        if MODE_MONITOR_TIME == "None" and MODE_MONITOR_MEMORY == "None":
            return []
        
        summary = []
        
        if MODE_MONITOR_TIME == "Standard":
            summary.append("=== Performance Timing ===")
            for op_name, metrics in self.metrics.items():
                if "duration_ms" in metrics:
                    duration_str = format_timing(metrics["duration_ms"])
                    summary.append(f"{op_name}: {duration_str}")
        
        if MODE_MONITOR_MEMORY == "Light":
            summary.append("=== Memory Usage ===")
            current_memory = get_memory_usage()
            memory_growth = current_memory - self.start_memory
            summary.append(f"Current Memory: {current_memory}MB")
            summary.append(f"Memory Growth: {memory_growth:+.2f}MB")
            
            # Show memory-intensive operations
            memory_ops = [(op, m) for op, m in self.metrics.items() 
                        if "memory_delta_mb" in m and abs(m["memory_delta_mb"]) > 1.0]
            if memory_ops:
                summary.append("Memory-intensive operations:")
                for op_name, metrics in memory_ops:
                    delta = metrics["memory_delta_mb"]
                    summary.append(f"  {op_name}: {delta:+.2f}MB")
        
        return summary

# Replace RetryRunnable usage with simple retry logic:
def execute_step_with_retry(tool_impl, step_context, max_attempts=3):
    for attempt in range(max_attempts):
        try:
            return tool_impl(step_context)
        except Exception as e:
            if attempt == max_attempts - 1:
                raise e
            time.sleep(1)
            
# Pydantic Models for Structured Outputs
class TaskClassificationResult(BaseModel):
    """Result of task classification."""
    task_type: str = Field(description="The identified task type")
    confidence: float = Field(description="Confidence score 0-1")
    reasoning: str = Field(description="Explanation for classification")


class ParameterExtractionResult(BaseModel):
    """Parameters extracted from task input."""
    parameters: Dict[str, Any] = Field(description="Extracted parameters")
    context: Dict[str, Any] = Field(description="Additional context", default_factory=dict)


class ExecutionResult(BaseModel):
    """Result of plan execution."""
    success: bool = Field(description="Whether execution succeeded")
    outputs: Dict[str, Any] = Field(description="Step outputs", default_factory=dict)
    errors: List[str] = Field(description="Any errors encountered", default_factory=list)
    logs: List[str] = Field(default_factory=list)


class ValidationResult(BaseModel):
    """Result of answer validation."""
    is_valid: bool = Field(description="Whether answer is valid")
    confidence: float = Field(description="Validation confidence 0-1")
    issues: List[str] = Field(description="Validation issues", default_factory=list)


class ParsedAnswer(BaseModel):
    """Final parsed answer."""
    answer: str = Field(description="The extracted answer")
    reasoning: str = Field(description="How answer was derived")
    metadata: Dict[str, Any] = Field(description="Additional metadata", default_factory=dict)


class AgentFramework:
    """
    Generic, domain-agnostic agent framework using LangChain LCEL.
    All domain-specific logic is externalized to configuration files.
    """
    
    def __init__(
        self, 
        config_dir = None,
        model_name: str = DEFAULT_MODEL_NAME,
        config_data = None
    ):
        if config_dir is None:
            # Point to the config directory within the package
            config_dir = str(Path(__file__).parent.parent / "config")
        self.config_dir = Path(config_dir)
        self.model_name = model_name
        self.llm = create_langchain_model(
            model_name=model_name,
            config_data=config_data
        )
        
        if DEBUG_PRINT:
            log_event(f"LangChain LCEL: Using model: {model_name}")
        
        # Load configurations
        self.tasks_config = self._load_config("tasks.json")
        self.plans_config = self._load_config("plans.json")
        self.tools_registry = self._load_tools_registry()
        self.config_data = config_data
        # Initialize LCEL components
        self._setup_pipelines()
    
    def _load_config(self, filename: str) -> Dict[str, Any]:
        # Load configuration from JSON file
        config_path = self.config_dir / filename
        if not config_path.exists():
            return {}
        
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _load_tools_registry(self) -> Dict[str, Any]:
        # Load all tool configurations from tools directory
        tools_dir = self.config_dir / "tools"
        registry = {}
        
        if tools_dir.exists():
            for tool_file in tools_dir.glob("*.json"):
                with open(tool_file, 'r') as f:
                    tool_config = json.load(f)
                    registry.update(tool_config)
        
        return registry
    
    def _extract_classification_patterns(self) -> Dict[str, Any]:
        # Extract only the classification-relevant parts of task config
        classification_data = {}
        for task_type, config in self.tasks_config.items():
            classification_data[task_type] = {
                "description": config.get("description", ""),
                "patterns": config.get("patterns", [])
            }
        
        return classification_data

    def _get_task_config(self, task_type: str, use_fallback: bool = False) -> Dict[str, Any]:
        """Get task configuration with optional fallback handling."""
        # Try exact match first
        if task_type in self.tasks_config:
            return self.tasks_config[task_type]
        # If no exact match and fallback is disabled, return empty
        if not use_fallback:
            return {}
        # Normalize function: remove separators and make lowercase
        def normalize(text):
            return text.replace("-", "").replace("_", "").replace(" ", "").lower()
        
        normalized_input = normalize(task_type)
        # Find matching config key using normalized comparison
        for config_key in self.tasks_config.keys():
            if normalize(config_key) == normalized_input:
                if DEBUG_PRINT:
                    log_event(f"Task type normalized match: '{task_type}' -> '{config_key}'")
                return self.tasks_config[config_key]
        # No match found
        return {}

    def _setup_pipelines(self):
        """Setup LangChain LCEL pipelines for each step."""
        
        # Task Classification Pipeline
        classification_parser = PydanticOutputParser(pydantic_object=TaskClassificationResult)
        
        # Debug wrapper for prompt formatting
        def classification_prompt_wrapper(inputs):
            debug_log("Task Classification - Input Data", {
                "task": inputs.get("task", ""),
                "available_task_types": list(self.tasks_config.keys())
            }, "CLASSIFICATION")
            formatted_prompt = classification_prompt.format(**inputs)
            debug_log("Task Classification - Formatted Prompt", formatted_prompt, "CLASSIFICATION")
            return formatted_prompt
        
        # Debug wrapper for LLM call
        def classification_llm_wrapper(prompt_text):
            debug_log("Task Classification - Sending to LLM", f"Prompt length: {len(prompt_text)} chars", "CLASSIFICATION", False)
            try:
                result = invoke_llm_with_config(self.llm, prompt_text, self.model_name)
                # Experiment: try overwrite to check effects
                # model_name_ow = "google/gemma-2-2b-it"
                # llm_ow = create_langchain_model(model_name_ow)
                # result = invoke_llm_with_config(llm_ow, prompt_text, model_name_ow)
                
                raw_content = result.content if hasattr(result, 'content') else str(result)
                debug_log("Task Classification - LLM Response", raw_content, "CLASSIFICATION")
                return result
            except Exception as e:
                debug_log("Task Classification - LLM Error", str(e), "CLASSIFICATION")
                raise
        
        # Debug wrapper for parsing
        def classification_parser_wrapper(llm_output):
            raw_content = llm_output.content if hasattr(llm_output, 'content') else str(llm_output)
            debug_log("Task Classification - Raw Content to Parse", raw_content, "CLASSIFICATION")
            try:
                # Get JSON string for PydanticOutputParser
                json_content = extract_json_string_from_llm_response(raw_content)
                parsed_result = classification_parser.parse(json_content)
                debug_log("Task Classification - Parsed Result", {
                    "task_type": parsed_result.task_type,
                    "confidence": parsed_result.confidence,
                    "reasoning": parsed_result.reasoning
                }, "CLASSIFICATION")
                return parsed_result
            
            except Exception as e:
                debug_log("Task Classification - Parsing Error", {
                    "error": str(e),
                    "raw_content": raw_content
                }, "CLASSIFICATION")
                # Fallback: get dict and create TaskClassificationResult manually
                try:
                    json_data = extract_json_dict_from_llm_response(
                        response_text=raw_content,
                        fallback_parser=_parse_classification_fallback,
                        required_fields=["task_type"]
                    )
                    return TaskClassificationResult(
                        task_type=json_data.get("task_type", "unknown"),
                        confidence=json_data.get("confidence", 0.0),
                        reasoning=json_data.get("reasoning", "Failed to parse with PydanticOutputParser")
                    )
                except Exception as fallback_error:
                    raise Exception(f"Could not parse JSON from LLM response. Original error: {e}, Fallback error: {fallback_error}")
        
        classification_prompt = PromptTemplate(
            template="""
            Classify the following task based on the patterns provided.
            
            Task Patterns: {task_patterns}
            Input Task: {task}
            
            You must respond with a valid JSON object containing exactly these fields:
            - task_type: the matching task type from the patterns
            - confidence: a number between 0 and 1
            - reasoning: explanation for your choice
            
            Example response:
            {{"task_type": "gene_alias", "confidence": 0.9, "reasoning": "This asks for a gene symbol"}}
            """,
            input_variables=["task_patterns", "task"]
        )
        
        self.classification_pipeline = (
            RunnablePassthrough.assign(
                task_patterns=lambda x: json.dumps(self._extract_classification_patterns(), indent=2)
            )
            | classification_prompt_wrapper # classification_prompt
            | classification_llm_wrapper # self.llm
            | classification_parser_wrapper # classification_parser
        )
            
        # Parameter Extraction Pipeline
        parameter_parser = PydanticOutputParser(pydantic_object=ParameterExtractionResult)
        parameter_prompt = PromptTemplate(
            template="""
            Extract parameters from the task based on the task configuration.
            
            Task Type: {task_type}
            Task Config: {task_config}
            Original Task: {original_task}
            
            {format_instructions}
            """,
            input_variables=["task_type", "task_config", "original_task"],
            partial_variables={"format_instructions": parameter_parser.get_format_instructions()}
        )
        
        self.parameter_pipeline = (
            parameter_prompt
            | self.llm
            | parameter_parser
        )
        
        # Answer Parsing Pipeline
        answer_parser = PydanticOutputParser(pydantic_object=ParsedAnswer)
        
        def answer_llm_wrapper(prompt_text):
            if hasattr(prompt_text, 'text'):
                prompt_text = prompt_text.text
            elif not isinstance(prompt_text, str):
                prompt_text= str(prompt_text)
            else:
                prompt_text = prompt_text
                
            debug_log("Answer Parsing - LLM Input", prompt_text, "PARSING")
            result = invoke_llm_with_config(self.llm, prompt_text, self.model_name)

            raw_response = result.content if hasattr(result, 'content') else str(result)
            debug_log("Answer Parsing - LLM Output", raw_response, "PARSING")
            return result

        def answer_parser_wrapper(llm_output):
            raw_content = llm_output.content if hasattr(llm_output, 'content') else str(llm_output)
            
            try:
                # Get JSON string for PydanticOutputParser
                json_content = extract_json_string_from_llm_response(raw_content)
                result = answer_parser.parse(json_content)
                debug_log("Answer Parser Success", result.dict(), "PARSING")
                return result
                
            except Exception as e:
                debug_log("Answer Parser Failed", f"Error: {e}\nRaw content: '{raw_content}'", "PARSING")
                
                # Fallback: get dict and create ParsedAnswer manually
                try:
                    json_data = extract_json_dict_from_llm_response(
                        response_text=raw_content,
                        fallback_parser=_parse_answer_fallback,
                        required_fields=["answer"]
                    )
                    return ParsedAnswer(
                        answer=json_data.get("answer", ""),
                        reasoning=json_data.get("reasoning", ""),
                        metadata=json_data.get("metadata", {})
                    )
                except Exception as fallback_error:
                    raise Exception(f"Could not parse JSON from LLM response. Original error: {e}, Fallback error: {fallback_error}")

            # - Do NOT include thinking steps, reasoning process, or <think> tags
            # - Do NOT explain your process - just extract the answer
        answer_prompt = PromptTemplate(
            template="""
            Parse the final answer from the execution results.
            
            Original Task: {original_task}
            Execution Results: {execution_results}
            
            answer formatting requirements: {answer_format}
            
            CRITICAL REQUIREMENTS:
            - Return ONLY a JSON object with no additional text, explanations, or markdown formatting
            - Do NOT include JavaScript-style comments (// or /* */) or trailing commas in the JSON
            - Use proper JSON syntax only
    
            You must return a valid JSON object with these exact fields:
            - answer: the extracted answer (following the requirements above)
            - reasoning: how the answer was derived
            - metadata: any additional information (can be empty object)
            
            Example response:
            {{"answer": "chr15", "reasoning": "Found in chromosome location field", "metadata": {{}}}}
            
            Your response:
            """,
            input_variables=["original_task", "execution_results", "answer_format"]
        )
        
        self.answer_pipeline = (
            answer_prompt
            | answer_llm_wrapper      # self.llm
            | answer_parser_wrapper   # answer_parser
        )
        
        # Validation Pipeline
        validation_parser = PydanticOutputParser(pydantic_object=ValidationResult)
        validation_prompt = PromptTemplate(
            template="""
            Validate the parsed answer against the original task.
            
            Original Task: {original_task}
            Parsed Answer: {parsed_answer}
            Validation Criteria: {validation_criteria}
            
            {format_instructions}
            """,
            input_variables=["original_task", "parsed_answer", "validation_criteria"],
            partial_variables={"format_instructions": validation_parser.get_format_instructions()}
        )
        
        self.validation_pipeline = (
            validation_prompt
            | self.llm
            | validation_parser
        )
    
    def _smart_truncate_execution_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Intelligently truncate execution results based on configuration."""
        config = FINAL_PARSING_CONFIG
        truncated = {}
        total_chars = 0
        
        # Step 1: Always preserve critical fields (never truncate these)
        for field in config["preserve_fields"]:
            if field in results:
                truncated[field] = results[field]
                if isinstance(results[field], str):
                    total_chars += len(results[field])
        
        # Step 2: Handle priority fields (truncate only if necessary)
        for field in config["priority_fields"]:
            if field in results and field not in config["preserve_fields"]:
                value = results[field]
                if isinstance(value, str):
                    field_limit = config["field_limits"].get(field, 1000)
                    
                    if len(value) > field_limit:
                        # Truncate but try to preserve meaningful content
                        truncated_value = self._intelligent_truncate(value, field_limit, field)
                        truncated[field] = truncated_value
                        total_chars += len(truncated_value)
                    else:
                        truncated[field] = value
                        total_chars += len(value)
                else:
                    truncated[field] = value
        
        # Step 3: Handle remaining fields with stricter limits
        remaining_budget = max(0, config["max_total_chars"] - total_chars)
        
        for key, value in results.items():
            if key not in truncated:  # Not already processed
                if isinstance(value, str):
                    field_limit = config["field_limits"].get(key, 500)
                    
                    # Use smaller of field limit or remaining budget
                    actual_limit = min(field_limit, remaining_budget // max(1, len([k for k in results.keys() if k not in truncated])))
                    
                    if len(value) > actual_limit and actual_limit > 0:
                        truncated_value = self._intelligent_truncate(value, actual_limit, key)
                        truncated[key] = truncated_value
                        remaining_budget -= len(truncated_value)
                    elif actual_limit > 0:
                        truncated[key] = value
                        remaining_budget -= len(value)
                    else:
                        # No budget left, create minimal summary
                        truncated[key] = f"[{key} truncated - {len(value)} chars]"
                else:
                    truncated[key] = value
        
        return truncated

    def _intelligent_truncate(self, text: str, limit: int, field_type: str) -> str:
        """Intelligently truncate text based on field type."""
        if len(text) <= limit:
            return text
        
        # Different truncation strategies based on field type
        if field_type == "document":
            # For documents, try to keep the beginning and end
            if limit > 200:
                head_size = int(limit * 0.7)
                tail_size = limit - head_size - 50  # Leave space for truncation message
                return (text[:head_size] + 
                        f"\n... [truncated {len(text) - limit} chars] ...\n" + 
                        text[-tail_size:])
            else:
                return text[:limit-20] + f"... [{len(text)} chars]"
        
        elif field_type in ["blast_result", "api_response"]:
            # For structured data, keep the beginning (usually contains key info)
            return text[:limit-30] + f"... [truncated from {len(text)} chars]"
        
        elif field_type == "logs":
            # For logs, keep the end (most recent info)
            return f"... [truncated {len(text) - limit} chars] ..." + text[-(limit-50):]
        
        else:
            # Default: simple truncation
            return text[:limit-20] + f"... [truncated]"
    
    def get_truncation_stats(self, original_results: Dict[str, Any], truncated_results: Dict[str, Any]) -> Dict[str, Any]:
        """Get statistics about truncation for debugging."""
        stats = {
            "original_total_chars": sum(len(str(v)) for v in original_results.values()),
            "truncated_total_chars": sum(len(str(v)) for v in truncated_results.values()),
            "fields_truncated": [],
            "fields_preserved": []
        }
        
        for key in original_results:
            if key in truncated_results:
                orig_len = len(str(original_results[key]))
                trunc_len = len(str(truncated_results[key]))
                
                if orig_len != trunc_len:
                    stats["fields_truncated"].append({
                        "field": key,
                        "original_chars": orig_len,
                        "truncated_chars": trunc_len,
                        "reduction": orig_len - trunc_len
                    })
                else:
                    stats["fields_preserved"].append(key)
        
        return stats

    def classify_task(self, task: str, monitor: PerformanceMonitor, cost_tracker=None) -> TaskClassificationResult:
        """Classify task using LCEL pipeline."""
        monitor.start_timer("task_classification")
        input_text = task
        result = self.classification_pipeline.invoke({"task": task})
        output_text = str(result)
        # Track cost if tracker provided
        if cost_tracker:
            cost_tracker.log_call(self.model_name, input_text, output_text)
        monitor.end_timer("task_classification")
        return result
    
    def extract_parameters(
        self, 
        task: str, 
        task_type: str, 
        task_config: Dict[str, Any],
        monitor: PerformanceMonitor,
        cost_tracker=None
    ) -> ParameterExtractionResult:
        """Extract parameters using LCEL pipeline."""
        monitor.start_timer("parameter_extraction")
        input_data = {
            "task_type": task_type,
            "task_config": json.dumps(task_config, indent=2),
            "original_task": task
        }
        input_text = f"Task: {task}, Type: {task_type}"
        result = self.parameter_pipeline.invoke(input_data)
        output_text = str(result)
        # Track cost if tracker provided
        if cost_tracker:
            cost_tracker.log_call(self.model_name, input_text, output_text)
        
        monitor.end_timer("parameter_extraction")
        return result
    
    def execute_plan(
        self, 
        plan: Dict[str, Any], 
        initial_context: Dict[str, Any],
        monitor: PerformanceMonitor
    ) -> ExecutionResult:
        """Execute a plan using configured tools."""
        outputs = {}
        try:
            context = initial_context.copy()
            tool_logs = []
            accumulated_api_urls = []
            
            monitor.start_timer("plan_execution_total")
            
            for step in plan.get("steps", []):
                step_name = step["function"]
                step_inputs = step.get("inputs", [])
                step_outputs = step.get("outputs", [])
                
                # Get tool implementation
                if step_name not in self.tools_registry:
                    raise ValueError(f"Tool '{step_name}' not found in registry")
                
                tool_config = self.tools_registry[step_name]
                try:
                    tool_impl = self._get_tool_implementation(tool_config)
                except Exception as e:
                    raise ValueError(f"Failed to get implementation for tool '{step_name}': {e}")
                
                # Prepare step inputs from context
                step_context = {}
                for input_key in step_inputs:
                    if input_key in context:
                        step_context[input_key] = context[input_key]
                
                step_context["api_urls"] = accumulated_api_urls.copy()
                step_context["tool_config"] = tool_config
                step_context["cost_tracker"] = context.get("cost_tracker")
                step_context["model_name"] = self.model_name
                
                # Execute step with monitoring
                monitor.start_timer(f"step_{step_name}")
                step_result = execute_step_with_retry(tool_impl, step_context)
                monitor.end_timer(f"step_{step_name}")
                
                if step_result and isinstance(step_result, dict) and 'logs' in step_result:
                    tool_logs.extend(step_result['logs'])
                
                if step_result and isinstance(step_result, dict) and 'api_urls' in step_result:
                    accumulated_api_urls.extend(step_result['api_urls'])
                    
                # Update context with outputs
                if isinstance(step_result, dict):
                    for output_key in step_outputs:
                        if output_key in step_result:
                            context[output_key] = step_result[output_key]
                            outputs[output_key] = step_result[output_key]
                else:
                    # Single output
                    if step_outputs:
                        context[step_outputs[0]] = step_result
                        outputs[step_outputs[0]] = step_result
            
            monitor.end_timer("plan_execution_total")
            
            outputs["api_urls"] = accumulated_api_urls
            
            return ExecutionResult(success=True, outputs=outputs, logs=tool_logs)
            
        except Exception as e:
            monitor.end_timer("plan_execution_total")
            return ExecutionResult(
                success=False, 
                outputs=outputs if 'outputs' in locals() else {}, 
                errors=[str(e)]
            )
    
    def validate_answer(
        self, 
        original_task: str, 
        parsed_answer: ParsedAnswer,
        validation_criteria: Dict[str, Any],
        monitor: PerformanceMonitor
    ) -> ValidationResult:
        """Validate answer using LCEL pipeline."""
        monitor.start_timer("answer_validation")
        result = self.validation_pipeline.invoke({
            "original_task": original_task,
            "parsed_answer": parsed_answer.dict(),
            "validation_criteria": json.dumps(validation_criteria, indent=2)
        })
        monitor.end_timer("answer_validation")
        return result
    
    def parse_final_answer(
        self, 
        original_task: str, 
        execution_results: Dict[str, Any],
        answer_format: Dict[str, Any],
        monitor: PerformanceMonitor,
        cost_tracker=None
    ) -> ParsedAnswer:
        """Parse final answer using LCEL pipeline."""
        monitor.start_timer("final_answer_parsing")
        
        remove_document_before_parsing = False # experiment to reduce size but not a great idea
        if remove_document_before_parsing:
            execution_results.pop("document", None)  # Reduce size

        # Apply smart truncation before sending to LLM
        truncated_results = self._smart_truncate_execution_results(execution_results)
        
        if DEBUG_PRINT:
            stats = self.get_truncation_stats(execution_results, truncated_results)
            if stats['original_total_chars'] != stats['truncated_total_chars']:
                log_event(f"Truncation applied: {stats['original_total_chars']} -> {stats['truncated_total_chars']} chars")
                if stats['fields_truncated']:
                    log_event(f"Truncated fields: {[f['field'] for f in stats['fields_truncated']]}")
                    
        input_text = f"Task: {original_task}, Results: {json.dumps(truncated_results, indent=2)}"
        result = self.answer_pipeline.invoke({
            "original_task": original_task,
            "execution_results": json.dumps(truncated_results, indent=2),  # Use truncated results
            "answer_format": json.dumps(answer_format, indent=2)
        })
        output_text = str(result)
        # Track cost using the actual data sent to LLM
        if cost_tracker:
            cost_tracker.log_call(self.model_name, input_text, output_text)
        
        monitor.end_timer("final_answer_parsing")
        return result
    
    def _get_tool_implementation(self, tool_config: Dict[str, Any]):
        """Get tool implementation from config."""
        impl_type = tool_config.get("implementation_type", "function")
        
        if impl_type == "function":
            # Dynamic import of function
            module_path = tool_config["module"]
            function_name = tool_config["function"]
            
            # Import the module dynamically
            import importlib
            module = importlib.import_module(module_path)
            return getattr(module, function_name)
        
        elif impl_type == "api":
            # Create API wrapper function
            def api_wrapper(context):
                raise NotImplementedError("API wrapper not implemented yet")
            return api_wrapper
        
        else:
            raise ValueError(f"Unknown implementation type: {impl_type}")
    
    def process_task(
        self, 
        task: str, 
        expected_answer: Optional[str] = None,
        **kwargs
    ) -> List[Any]:
        """
        Main entry point - process a task through the complete pipeline.
        Returns format compatible with existing codebase:
        [question, answer, final_answer, logs, api_calls, elapsed_time]
        """
        start_time = time.time()
        cost_tracker = UniversalCostTracker()
        kwargs['cost_tracker'] = cost_tracker
        
        logs = []
        api_calls = []
        
        # Initialize performance monitoring
        monitor = PerformanceMonitor()
        monitor.start_timer("total_processing")
        
        try:
            # Step 1: Task Classification
            logs.append("Stage 1: Task Classification")
            classification_result = self.classify_task(task, monitor, cost_tracker)
            logs.append(f"Task classified as: {classification_result.task_type}")
            
            if DEBUG_PRINT:
                log_event("="*10 + " Stage 1: Task Classification " + "="*10)
                log_event(f"Task classified as: {classification_result.task_type}")
                log_event(f"Available tasks: {list(self.tasks_config.keys())}")
                
            # Get task configuration
            # task_config = self.tasks_config.get(classification_result.task_type, {})
            task_config = self._get_task_config(classification_result.task_type, use_fallback=True)
            
            config_data = self.config_data
            if config_data and config_data.get("request", {}).get("type") == "task_classification":
                elapsed_time = time.time() - start_time
                raw_classification_response = str(classification_result)
                logs.append("=== TASK CLASSIFICATION DETAILS ===")
                logs.append(f"Raw LLM Response: {raw_classification_response}")
                logs.append(f"Parsed Task Type: {classification_result.task_type}")
                logs.append(f"Confidence: {classification_result.confidence}")
                logs.append(f"Reasoning: {classification_result.reasoning}")
                if task_config:
                    logs.append(f"Task Config: {json.dumps(task_config, indent=2)}")
                    
                return [
                    task,
                    expected_answer,
                    classification_result.task_type,
                    logs,
                    [],
                    elapsed_time
                ]
                
            if not task_config:
                raise ValueError(f"No configuration found for task type: {classification_result.task_type}")
            
            # Step 2: Parameter Extraction (leave it empty as we have specific tools for this)
            USE_FRAMEWORK_PARAMETER_EXTRACTION = False
            if USE_FRAMEWORK_PARAMETER_EXTRACTION:
                logs.append("Stage 2: Parameter Extraction")
                parameter_result = self.extract_parameters(task, classification_result.task_type, task_config, monitor, cost_tracker)
                logs.append(f"Parameters extracted: {list(parameter_result.parameters.keys())}")
            else:
                parameter_result = ParameterExtractionResult(
                    parameters={},  # Empty - tools will extract what they need
                    context={}
                )

            # Step 2/3: Plan Retrieval and Execution
            logs.append("Stage 2: Plan Retrieval")
            plan_name = task_config.get("plan", classification_result.task_type)
            plan = self.plans_config.get(plan_name, {})
            
            if not plan:
                raise ValueError(f"No plan found for: {plan_name}")
            logs.append(f"Plan retrieved: {plan_name}")
            if DEBUG_PRINT:
                log_event("="*10 + " Stage 2: Plan Retrieval " + "="*10,)
                log_event(f"Plan retrieved: {plan_name}")
                
            logs.append("Stage 3: Plan Execution")
            step_names = [step["function"] for step in plan.get("steps", [])]
            logs.append(f"Plan of {len(step_names)} steps: {' --> '.join(step_names)}")
                
            initial_context = {
                "task": task,
                "parameters": parameter_result.parameters,
                "task_type": classification_result.task_type,
                "classification_confidence": classification_result.confidence,
                "llm": self.llm,
                "cost_tracker": cost_tracker,
                **parameter_result.context
            }
                
            execution_result = self.execute_plan(plan, initial_context, monitor)
            logs.extend(execution_result.logs)  # Add tool logs to main logs
            
            # Extract API URLs from execution results
            if "api_urls" in execution_result.outputs:
                api_calls = execution_result.outputs["api_urls"]
                logs.append(f"Total API calls made: {len(api_calls)}")
            
            if not execution_result.success:
                raise ValueError(f"Plan execution failed: {execution_result.errors}")
            
            if DEBUG_PRINT:
                log_event("="*10 + " Stage 3: Plan Execution " + "="*10,)
                log_event(f"Plan of {len(step_names)} steps: {' --> '.join(step_names)}")
                log_event(execution_result.logs)
                
            # Step 4: Aggregate Parsing as a Generalist (e.g. Format the answer with common sense)
            logs.append("Stage 4: Answer Format Parsing")
            answer_format = task_config.get("answer_format", {})
                
            # Extract the raw answer from tool outputs BEFORE final parsing
            raw_answer_before_parsing = None
            if isinstance(execution_result.outputs, dict) and 'answer' in execution_result.outputs:
                raw_answer_before_parsing = execution_result.outputs['answer']
            else:
                raw_answer_before_parsing = str(execution_result.outputs)
            
            if DEBUG_PRINT:
                log_event("="*10 + " Stage 4: Answer Format Parsing " + "="*10,)
                log_event(f"Raw answer before parsing: {raw_answer_before_parsing}")
                
            # at task level, we can define a minimum length to parse at framework level
            # 0 (default) means always parse if answer_format is defined
            # 99999999 means never parse, always pass-through
            # 100 means parse only if raw answer is at least 100 characters (eg when LLM goes into reasoning mode)
            min_length_to_parse = task_config.get("min_length_to_parse_answer", 0)
            raw_answer_length = len(str(raw_answer_before_parsing)) if raw_answer_before_parsing else 0
            
            if answer_format and raw_answer_length >= min_length_to_parse:
                logs.append(f"Answer length ({raw_answer_length}) >= threshold ({min_length_to_parse}) - applying framework parsing...")
                try:
                    parsed_answer = self.parse_final_answer(task, execution_result.outputs, answer_format, monitor, cost_tracker)
                    # Assume parsed_answer has an answer attribute in this case
                    final_answer = parsed_answer.answer

                except Exception as json_error:
                    # GENERIC FALLBACK: Just use the raw tool output
                    logs.append(f"❌ Framework JSON parsing failed: {str(json_error)}")
                    logs.append("🔄 Falling back to raw tool output...")
                    parsed_answer = ParsedAnswer(
                        answer=str(raw_answer_before_parsing),
                        reasoning="JSON parsing failed, using raw tool output",
                        metadata={"fallback_used": True, "json_error": str(json_error)}
                    )
                    final_answer = parsed_answer.answer
                    
                # Compare raw tool output vs final parsed answer
                if DEBUG_PRINT:
                    if str(raw_answer_before_parsing) == str(final_answer):
                        logs.append("SUCCESS: PARSING IMPACT: Raw tool output and final parsed answer are IDENTICAL - final parsing may be unnecessary")
                        log_event("SUCCESS: PARSING IMPACT: Raw tool output and final parsed answer are IDENTICAL - final parsing may be unnecessary")
                    else:
                        logs.append("WARNING: PARSING IMPACT: Raw tool output and final parsed answer are DIFFERENT - final parsing is adding value")
                        logs.append(f"   Raw tool output: '{raw_answer_before_parsing}'")
                        logs.append(f"   Final parsed:    '{final_answer}'")
                        log_event("WARNING: PARSING IMPACT: Raw tool output and final parsed answer are DIFFERENT - final parsing is adding value")
                        log_event(f"   Raw tool output: '{raw_answer_before_parsing}'")
                        log_event(f"   Final parsed:    '{final_answer}'")
                        
            else:
                logs.append("Pass-through final answer...")
                parsed_answer = execution_result.outputs
                if isinstance(parsed_answer, dict) and 'answer' in parsed_answer:
                    final_answer = parsed_answer['answer']  # Access the answer as a dictionary key
                else:
                    final_answer = parsed_answer
                    
                # For pass-through cases, they're identical by definition
                if DEBUG_PRINT:
                    logs.append("SUCCESS: PARSING IMPACT: No final parsing applied - using raw tool output directly")
                    log_event("SUCCESS: PARSING IMPACT: No final parsing applied - using raw tool output directly")

            # Step 5: Validation (optional. TO-DO)
            validation_criteria = task_config.get("validation", {})
            if validation_criteria:
                logs.append("Validating answer...")
                # Ensure parsed_answer is a ParsedAnswer instance
                if not isinstance(parsed_answer, ParsedAnswer):
                    try:
                        parsed_answer_obj = ParsedAnswer.parse_obj(parsed_answer)
                    except Exception as e:
                        logs.append(f"Failed to convert parsed_answer to ParsedAnswer: {e}")
                        parsed_answer_obj = ParsedAnswer(answer=str(parsed_answer), reasoning="", metadata={})
                else:
                    parsed_answer_obj = parsed_answer
                validation_result = self.validate_answer(task, parsed_answer_obj, validation_criteria, monitor)
                logs.append(f"Validation result: {validation_result.is_valid}")

            monitor.end_timer("total_processing")
            elapsed_time = time.time() - start_time

            # Add performance metrics to logs
            performance_summary = monitor.get_performance_summary()
            if performance_summary:
                logs.extend(performance_summary)

            cost_summary = cost_tracker.get_output_summary()
            logs.append("")
            logs.extend(cost_summary)
            # Return in expected format
            return [
                task,                # question
                expected_answer,     # answer
                final_answer,        # final_answer - use our extracted value
                logs,                # logs
                api_calls,           # api_calls
                elapsed_time         # elapsed_time
            ]
            
        except Exception as e:
            monitor.end_timer("total_processing")
            elapsed_time = time.time() - start_time
            logs.append(f"Error: {str(e)}")
            
            # Add performance metrics even on error
            performance_summary = monitor.get_performance_summary()
            if performance_summary:
                logs.extend(performance_summary)
            
            return [
                task,
                expected_answer,
                f"Error: {str(e)}",
                logs,
                api_calls,
                elapsed_time
            ]

# Fallback Reg-Ex parser for task classification when JSON parsing fails
def _parse_classification_fallback(response_text):
    # More flexible patterns to handle different formatting
    task_type_patterns = [
        r'\*{1,3}\s*Task Type\s*:?\s*\*{1,3}\s*(\w+)',  # **Task Type:** or *Task Type* or ***Task Type:***
        r'Task Type\s*:?\s*(\w+)',                      # Task Type: gene_alias (no stars)
        r'task_type\s*:?\s*["\']?(\w+)["\']?'
    ]
    
    confidence_patterns = [
        r'\*{1,3}\s*Confidence\s*:?\s*\*{1,3}\s*([\d.]+)',  # **Confidence:** 0.9
        r'Confidence\s*:?\s*([\d.]+)',                      # Confidence: 0.9 (no stars)
        r'confidence\s*:?\s*["\']?([\d.]+)["\']?',          # confidence: "0.9"
    ]
    
    # Try to extract task_type
    task_type = None
    for pattern in task_type_patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            task_type = match.group(1).lower()
            break
    
    # Try to extract confidence
    confidence = 0.5  # default
    for pattern in confidence_patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            confidence = float(match.group(1))
            break
    
    if task_type:
        return {
            "task_type": task_type,
            "confidence": confidence,
            "reasoning": "Extracted via manual parsing fallback"
        }
    
    raise ValueError("Could not parse task classification")

def _parse_answer_fallback(response_text):
    """Fallback parser for final answer when JSON parsing fails"""
    # Look for "Answer: XXX" pattern
    log_event(f"Fallback answer parser invoked {response_text}")
        # Look for "answer": "value" pattern (JSON style)
    answer_json_match = re.search(r'"answer"\s*:\s*"([^"]+)"', response_text, re.IGNORECASE)
    if answer_json_match:
        return {
            "answer": answer_json_match.group(1).strip(),
            "reasoning": "Extracted from answer field",
            "metadata": {}
        }
        
    answer_match = re.search(r'Answer:\s*(.+?)(?:\n|$)', response_text, re.IGNORECASE)
    if answer_match:
        return {
            "answer": answer_match.group(1).strip(),
            "reasoning": "Extracted from Answer: pattern",
            "metadata": {}
        }
    
    # If no "Answer:" pattern, try to extract the last meaningful line
    lines = [line.strip() for line in response_text.split('\n') if line.strip()]
    if lines:
        return {
            "answer": lines[-1],
            "reasoning": "Extracted last line as answer",
            "metadata": {}
        }
    
    raise ValueError("Could not parse answer")

# Main entry point function for integration
def answer_agent(
    question: str,
    answer: str,
    model_name: str = DEFAULT_MODEL_NAME,
    config_data = None,
    config_dir = None,
    **kwargs
) -> List[Any]:
    """
    Main entry point compatible with existing codebase.
    
    Args:
        question: The task/question to process
        answer: Expected answer (for validation)
        config_dir: Path to configuration directory
        **kwargs: Additional arguments
    
    Returns:
        [question, answer, final_answer, logs, api_calls, elapsed_time]
    """
    framework = AgentFramework(config_dir=config_dir, model_name=model_name, config_data=config_data)
    return framework.process_task(question, expected_answer=answer, **kwargs)

if __name__ == "__main__":
    # Example usage
    framework = AgentFramework()
    result = framework.process_task("What is the official gene symbol of SEP3?")
    log_event(json.dumps(result, indent=2))