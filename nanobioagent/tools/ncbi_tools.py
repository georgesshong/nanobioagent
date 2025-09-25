"""
NCBI-specific tool implementations for the generic agent framework.
These are domain-specific implementations that plug into the generic framework.
"""
DEFAULT_MODEL_NAME = "BadName to defend against fallback. In prod can set it back to default model" # "gpt-4o-mini" # Default model for LLM calls in this module
DEFAULT_RETMAX = 5  # Default number of records to return
USE_FALLBACK_WITHRULES = False
MAX_DOCUMENT_SIZE = 30000

import json
import time
import re
from typing import Dict, List, Any, Optional
from pathlib import Path
from .gene_utils import call_api, log_event
from ..core.model_utils import extract_json_dict_from_llm_response, create_langchain_model, invoke_llm_with_config, get_model_config, tokens_to_chars

CONFIG_DIR = Path(__file__).parent.parent / "config"
PATH_EXAMPLES = str(CONFIG_DIR / "examples" / "ncbi_examples.json")
# step function to infer parameters from task dictionaries: LLM-based and rule-based
def infer_parameters_from_task(context: Dict[str, Any]) -> Dict[str, Any]:
    """LLM-based parameter inference with step-specific examples."""
    # upgrade to FewShotChatMessagePromptTemplate API later if needed
    # Use the LLM from context if available
    llm = context.get("llm")
    model_name = context.get('model_name', DEFAULT_MODEL_NAME)
    if not llm:
        llm = create_langchain_model(model_name=model_name)

    task = context.get("task", "")
    task_type = context.get('task_type', '')
    tool_config = context.get("tool_config", {}) 
    general = tool_config.get("general_instructions", "")
    specific = tool_config.get("specific_instructions", {}).get(task_type, "")
    instruction_text = "\n".join(filter(None, [general, specific]))

    # Load step-specific examples
    examples = _load_step_examples("infer_parameters_from_task", task_type)
    examples_text = _build_examples_text(examples, input_keys=['task'], output_name='Parameters')
    
    prompt_text = f"""You are extracting parameters for NCBI API queries. Do NOT answer the question - only extract the API parameters."""
    prompt_text += f"""For task type '{task_type}', extract these parameters:"""
    prompt_text += """- database: "gene", "snp", "omim", or "nt"."""
    prompt_text += """- search_term: the main entity to search for"""
    prompt_text += """- retmax: number of results (default "5")"""
    prompt_text += """- retmode: return format (default "json")"""

    prompt_text += instruction_text
    '''
    prompt_text += """\nSpecial Rule for sequences with dots (clone for nucleotide sequences):
- IF and ONLY IF the search_term contains a literal dot character "." (period/full stop) followed by a one or two digit variant number (indicating a clone)
- THEN set database to "nuccore" AND search_term to the substring before the first dot
- Example 1: "AC104236.2" → database: "nuccore", search_term: "AC104236"
- Example 2: "CTB-180A7.6" → database: "nuccore", search_term: "CTB-180A7"
- This rule does NOT apply to: underscores "_", hyphens "-", or ENSEMBL IDs like "ENSG00000205403"."""
    '''
    logs = []
    if examples:
        prompt_text += examples_text
        prompt_text += "\nExtract parameters for this new task following the same pattern and return ONLY a JSON object."
        logs.extend([f"infer_parameters_from_task: Using few-shot with {len(examples)} examples for task type '{task_type}'"])
    else:
        prompt_text += "\nExtract parameters for this new task and return ONLY a JSON object."
        logs.extend([f"infer_parameters_from_task: Using zero-shot (no examples found for task type '{task_type}')"])
    
    prompt_text += f"\n[ACTUAL TASK TO PARSE - USE THIS, RETURN ONLY A JSON OBJECT]"
    prompt_text += f"\nTask: {task}"
    prompt_text += f"\nParameters:"
    log_event(prompt_text)

    try:
        # response = llm.invoke(prompt_text)
        response = invoke_llm_with_config(llm, prompt_text, model_name)
        
        # Add cost tracking
        cost_tracker = context.get('cost_tracker')
        if cost_tracker:
            model_name = context.get('model_name', 'unknown')
            response_text = response.content if hasattr(response, 'content') else str(response)
            cost_tracker.log_call(model_name, prompt_text, response_text)
        # Try to parse JSON response
        try:
            parameters = extract_json_dict_from_llm_response(response)
            # log_event(f"DEBUG: extract_json_dict_from_llm_response: {parameters}")
        except json.JSONDecodeError as e:
            # log_event(f"DEBUG: Raw LLM response content: {response.content}")
            # log_event(f"DEBUG: JSON decode error: {e}")
            if USE_FALLBACK_WITHRULES:
                # Only used for testing: Fallback to rule-based if JSON parsing fails
                log_event("Warning: LLM response not valid JSON, falling back to rule-based approach")
                log_event(f"infer_parameters_from_task response content: {response}")
                
                fallback_result = infer_parameters_from_task_withrules(context)        
                # Ensure fallback result includes our logs
                if isinstance(fallback_result, dict):
                    existing_logs = fallback_result.get('logs', [])
                    fallback_result['logs'] = logs + existing_logs
                    return fallback_result
                else:
                    return {"parameters": fallback_result, "logs": logs}
            else:
                raise  # Re-raise the JSONDecodeError if fallback is disabled
        
        if task_type in ['gene_alias']:
            parameters['orgn'] = 'homo sapiens'
        
        logs.append(f"infer_parameters_from_task: Extracted parameters: {parameters}")
        return {
            "parameters": parameters,
            "logs": logs
        }
        
    except Exception as e:
        if USE_FALLBACK_WITHRULES:
            log_event(f"Warning: LLM parameter extraction failed: {e}, falling back to rule-based")
            # Fallback to rule-based approach
            return infer_parameters_from_task_withrules(context)
        else:
            raise  # Re-raise the exception if fallback is disabled
    
# step function that performs NCBI esearch and returns ID list.
def get_esearch_idlist_from_param(context: Dict[str, Any]) -> Dict[str, Any]:
    parameters = context.get("parameters", {})
    
    database = parameters.get("database", "gene")
    search_term = parameters.get("search_term", "")
    retmax = parameters.get("retmax", DEFAULT_RETMAX)
    retmode = parameters.get("retmode", "json")
    orgn = parameters.get("orgn", "")
    
    if not search_term:
        raise ValueError("No search term provided")
    
    if orgn.lower() == "human" or orgn.lower() == "homo sapiens":
        if database == "gene":
            entrez_query = "%20NOT%20discontinued[prop]" # not discontinued property
            entrez_query = entrez_query + "%20AND%20Homo%20sapiens[orgn]"
            search_term = search_term + entrez_query
    
    url = f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db={database}&retmax={retmax}&retmode={retmode}&sort=relevance&term={search_term}'
    
    response = call_api(url)
    processed = _process_response_by_mode(response, retmode)
    
    # Extract idlist from response
    idlist = []
    if processed["actual_format"] == "json" and isinstance(processed["content"], dict):
        if "esearchresult" in processed["content"]:
            idlist = processed["content"]["esearchresult"].get("idlist", [])
    
    return {
        "idlist": idlist,
        "raw_response": processed["raw_response"],
        "database_used": database,
        "api_urls": [url],
        "logs": [
            f"get_esearch_idlist_from_param: URL called: {url}",
            f"get_esearch_idlist_from_param: Found {len(idlist)} IDs"
        ]
    }

# step function that performs NCBI esummary to get summary documents from ID list.
def get_esummary_doc_from_idlist(context: Dict[str, Any]) -> Dict[str, Any]:
    idlist = context.get("idlist", [])
    database_used = context.get("database_used", "gene")
    
    if not idlist:
        return {"document": "No IDs provided", "raw_response": "", "database_used": database_used, "api_urls": [], "logs": []}
    
    id_string = ",".join(idlist)
    retmax = max(DEFAULT_RETMAX, len(idlist))
    retmode = "json"  # esummary typically returns JSON
    
    url = f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db={database_used}&retmax={retmax}&retmode={retmode}&id={id_string}'
    
    response = call_api(url)
    processed = _process_response_by_mode(response, retmode)
    
    return {
        "document": processed["content"],
        "raw_response": processed["raw_response"],
        "database_used": database_used,
        "api_urls": [url],
        "logs": [
            f"get_esummary_doc_from_idlist: URL called: {url}",
            f"get_esummary_doc_from_idlist: Document retrieved: {len(processed['raw_response'])} characters",
            f"get_esummary_doc_from_idlist: Database used: {database_used}",
            f"get_esummary_doc_from_idlist: Retrieved {len(idlist)} IDs with retmax={retmax}"
        ]
    }

# step function that performs NCBI efetch to get full documents from ID list.
def get_efetch_doc_from_idlist(context: Dict[str, Any]) -> Dict[str, Any]:
    idlist = context.get("idlist", [])
    database_used = context.get("database_used", "gene")
    
    if not idlist:
        return {"document": "No IDs provided", "raw_response": ""}
    
    id_string = ",".join(idlist)
    retmax = max(DEFAULT_RETMAX, len(idlist))
    retmode = "text"  # efetch typically returns text format
    
    url = f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db={database_used}&retmax={retmax}&retmode={retmode}&id={id_string}'
    
    response = call_api(url)
    processed = _process_response_by_mode(response, retmode)
    
    return {
        "document": processed["content"],
        "raw_response": processed["raw_response"],
        "database_used": database_used,
        "api_urls": [url],
        "logs": [
            f"get_efetch_doc_from_idlist: URL called: {url}",
            f"get_efetch_doc_from_idlist: Document retrieved: {len(processed['raw_response'])} characters",
            f"get_efetch_doc_from_idlist: Document preview: {processed['content'][:3000]}...",
            f"get_efetch_doc_from_idlist: Database used: {database_used}",
            f"get_efetch_doc_from_idlist: Retrieved {len(idlist)} IDs with retmax={retmax}"
        ]
    }

# step function for SNP queries that go directly to esummary.
def get_snp_summary_from_id(context: Dict[str, Any]) -> Dict[str, Any]:
    parameters = context.get("parameters", {})
    search_term = parameters.get("search_term", "")
    retmax = parameters.get("retmax", DEFAULT_RETMAX)
    retmode = "json"
    
    if not search_term:
        raise ValueError("No search term provided")
    
    # Extract SNP ID (remove "rs" prefix if present)
    snp_id = search_term[2:] if search_term.startswith("rs") else search_term
    database_used = "snp"
    
    url = f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db={database_used}&retmax={retmax}&retmode={retmode}&id={snp_id}'
    
    response = call_api(url)
    processed = _process_response_by_mode(response, retmode)
    
    return {
        "idlist": [snp_id],
        "document": processed["content"],
        "raw_response": processed["raw_response"],
        "database_used": database_used,
        "api_urls": [url],
        "logs": [
            f"get_snp_summary_from_id: URL called: {url}",
            f"get_snp_summary_from_id: Retrieved SNP {snp_id}"
        ]
    }

# step function submitting BLAST query and return results (with caching handled by call_api)
def execute_blast_query(context: Dict[str, Any]) -> Dict[str, Any]:
    parameters = context.get("parameters", {})
    sequence = parameters.get("search_term", "")
    database = parameters.get("database", "nt")
    
    if not sequence:
        raise ValueError("No sequence provided for BLAST")
    
    logs = []
    # Step 1: Submit BLAST query (cached by call_api)
    put_url = f'https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi?CMD=Put&PROGRAM=blastn&MEGABLAST=on&DATABASE={database}&FORMAT_TYPE=XML&QUERY={sequence}&HITLIST_SIZE=5'
    
    response = call_api(put_url)
    response_text = response.decode('utf-8')
    logs.append(f"execute_blast_query: Put URL: {put_url}")
    
    # Extract RID
    rid_match = re.search(r'RID = (.*)\n', response_text)
    if not rid_match:
        raise ValueError("Could not extract RID from BLAST response")
    
    rid = rid_match.group(1)
    logs.append(f"execute_blast_query: RID: {rid}")
    
    # Step 2: Get results (cached by call_api, with automatic waiting)
    get_url = f'https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi?CMD=Get&FORMAT_TYPE=Text&RID={rid}'
    
    response = call_api(get_url)  # call_api handles waiting and caching
    result_text = response.decode('utf-8')
    logs.append(f"execute_blast_query: Get URL: {get_url}")
    logs.append(f"execute_blast_query: Sequence length: {len(sequence)}")
    
    return {
        "document": result_text,
        "raw_response": result_text,
        "database_used": database,
        "api_urls": [put_url, get_url],
        "logs": logs
    }
    
# BLAST-specific functions (for DNA alignment tasks)
def put_blast_query_for_rid(context: Dict[str, Any]) -> Dict[str, Any]:
    """Submit BLAST query and return RID."""
    parameters = context.get("parameters", {})
    sequence = parameters.get("search_term", "")
    database_used = "nt"
    if not sequence:
        raise ValueError("No sequence provided for BLAST")
    
    url = f'https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi?CMD=Put&PROGRAM=blastn&MEGABLAST=on&DATABASE={database_used}&FORMAT_TYPE=XML&QUERY={sequence}&HITLIST_SIZE=5'

    response = call_api(url)
    response_text = response.decode('utf-8')
    
    # Extract RID from response
    rid_match = re.search(r'RID = (.*)\n', response_text)
    if rid_match:
        rid = rid_match.group(1)
        return {
            "rid": rid,
            "raw_response": response_text,
            "database_used": database_used,
            "api_urls": [url],
            "logs": [
                f"put_blast_query_for_rid: URL called: {url}",
                f"put_blast_query_for_rid: Database used: {database_used}",
                f"put_blast_query_for_rid: Retrieved rid: {rid}"
            ]
        }
    else:
        raise ValueError("Could not extract RID from BLAST response")

def get_blast_doc_from_rid(context: Dict[str, Any]) -> Dict[str, Any]:
    """Get BLAST results using RID."""
    rid = context.get("rid", "")
    
    if not rid:
        raise ValueError("No RID provided")
    
    # Wait for BLAST to complete
    time.sleep(30)
    
    url = f'https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi?CMD=Get&FORMAT_TYPE=Text&RID={rid}'
    
    response = call_api(url)
    document = response.decode('utf-8')
    
    return {
        "document": document,
        "raw_response": document,
        "api_urls": [url],
        "logs": [
            f"get_blast_doc_from_rid: URL called: {url}"
            f"get_blast_doc_from_rid: Retrieved rid: {rid}"
        ]
    }
        

"""Few-shot LLM parsing for NCBI documents - step-specific learning."""
def parse_answer_from_doc(context: Dict[str, Any]) -> Dict[str, Any]:
    
    do_dynamic_truncation = True  # Set to False to use old MAX_DOCUMENT_SIZE logic
    
    # Use the LLM from context instead of creating a new one
    llm = context.get("llm")
    model_name = context.get('model_name', DEFAULT_MODEL_NAME)
    if not llm:
        # Fallback to creating one (shouldn't happen normally)
        llm = create_langchain_model(model_name=model_name)
    
    # document = context.get("document", "")
    document_raw = context.get("document", "")
    if isinstance(document_raw, dict):
        document = json.dumps(document_raw)
    else:
        document = document_raw
    
    task = context.get("task", "")
    api_urls = context.get("api_urls", [])
    logs = []
    if not document:
        return {"answer": "No document provided", "api_urls": [], "logs": ["parse_answer_from_doc: No document provided"]}
    
    original_doc_size = len(document)
    max_chars = 0
    if do_dynamic_truncation:
        # NEW LOGIC: Dynamic truncation based on model context window
        # Get model config for dynamic truncation
        model_config = get_model_config(model_name)
        context_window = model_config.get("context_window", 32000)
        max_tokens = model_config.get("max_tokens", 512)
        
        # Convert context window to characters (leave buffer for response)
        # Reserve space: context_window - max_tokens - safety_buffer
        safety_buffer = 0 # bigger the more conservative, but less accuracy
        safety_ratio = 1.10 # smaller the more conservative, but less accuracy
        available_tokens = context_window - max_tokens - safety_buffer  # 50 token safety buffer
        max_chars = int(tokens_to_chars(available_tokens, model_name) * safety_ratio)
        
        log_event(f"parse_answer_from_doc: Model: {model_name}, Context: {context_window} tokens, Max chars: {max_chars}")
        log_event(f"parse_answer_from_doc: do_dynamic_truncation: {do_dynamic_truncation}, safety_ratio: {safety_ratio}, safety_buffer: {safety_buffer}")
        log_event(f"parse_answer_from_doc: document type: {type(document)}")

    else:
        # TRUNCATE LARGE DOCUMENTS TO AVOID RATE LIMITS
        original_doc_size = len(document)
        if len(document) > MAX_DOCUMENT_SIZE:
            if 'blast' in api_urls[0].lower() if api_urls else False:
                document = _smart_truncate_blast_for_parsing(document, MAX_DOCUMENT_SIZE)
            else:
                document = document[:MAX_DOCUMENT_SIZE]
                document += "\n\n[Document truncated to avoid rate limits]"
        # Add truncation info to logs
        if original_doc_size > MAX_DOCUMENT_SIZE:
            logs.append(f"parse_answer_from_doc: Document truncated from {original_doc_size} to {len(document)} chars")
        
    # Determine task type for appropriate examples
    task_type = context.get('task_type', '')
    tool_config = context.get("tool_config", {}) 
    
    general = tool_config.get("general_instructions", "")
    specific = tool_config.get("specific_instructions", {}).get(task_type, "")
    instruction_text = "\n".join(filter(None, [general, specific]))

    # Load step-specific examples
    examples = _load_step_examples("parse_answer_from_doc", task_type)
    examples_text = _build_examples_text(examples)
    api_text = _build_api_text(api_urls)
    
    prompt_text = ""
    prompt_text += instruction_text
    
    if examples:
        prompt_text += examples_text
        prompt_text += "Now parse this new one following the same pattern:"
        logs = [f"parse_answer_from_doc: Using few-shot with {len(examples)} examples for task type '{task_type}'"]
    else:
        logs = [f"parse_answer_from_doc: Using zero-shot parsing (no examples found for task type '{task_type}')"]
    
    
    prompt_text += f"\nTask Type: {task_type}"
    prompt_text += f"\n[ACTUAL TASK TO PARSE - USE THIS]"
    prompt_text += f"{api_text}"
    
    num_char_prompt_so_far = len(prompt_text) # accumulated prompt
    if do_dynamic_truncation:
        # Calculate remaining budget for document
        max_document_chars = max(0, max_chars - num_char_prompt_so_far)
        # more conservative truncation
        max_document_chars = min(max_document_chars, MAX_DOCUMENT_SIZE)
        
        log_event(f"parse_answer_from_doc: Prompt so far: {num_char_prompt_so_far} chars, Document budget: {max_document_chars} chars")
        '''
        if len(document) > max_document_chars:
                logs.extend([f"parse_answer_from_doc: Truncation of doc from {len(document)} to {max_document_chars} chars"])
                if max_document_chars > 50:  # Only truncate if we have reasonable space
                    document = document[:max_document_chars-30]  # Leave space for truncation message
                    document += "\n[Document truncated...]"
                else:
                    document = "[Document too large for context]"
        '''
        if len(document) > max_document_chars:
            logs.extend([f"parse_answer_from_doc: Truncation of doc from {len(document)} to {max_document_chars} chars"])
            if max_document_chars > 50:  # Only truncate if we have reasonable space
                # Handle BLAST documents with smart truncation, others with simple truncation
                if 'blast' in api_urls[0].lower() if api_urls else False:
                    document = _smart_truncate_blast_for_parsing(document, max_document_chars)
                else:
                    document = document[:max_document_chars-30]  # Leave space for truncation message
                    document += "\n[Document truncated...]"
            else:
                document = "[Document too large for context]"
                
    # Add document to prompt
    prompt_text += f"\nDocument: {document}"
    prompt_text += f"\nTask: {task}"
    prompt_text += f"\n\nAnswer:"
    
    log_event("="*10 + " parse_answer_from_doc " + "="*10)
    log_event(prompt_text)
    log_event("="*30)
    log_event(f"parse_answer_from_doc: Instruction length: {len(instruction_text)} chars")
    log_event(f"parse_answer_from_doc: Examples length: {len(examples_text)} chars")
    log_event(f"parse_answer_from_doc: API text length: {len(api_text)} chars")
    log_event(f"parse_answer_from_doc: Document length: {len(document)} chars")
    log_event(f"parse_answer_from_doc: Prompt total length: {len(prompt_text)} chars")
    log_event("="*30)
    
    try:
        # response = llm.invoke(prompt_text)
        # response = invoke_llm_with_config(llm, prompt_text, model_name)
        # answer = response.content.strip()
        answer = invoke_llm_with_config(llm, prompt_text, model_name)
        cost_tracker = context.get('cost_tracker')
        if cost_tracker:
            model_name = context.get('model_name', 'unknown')
            cost_tracker.log_call(model_name, prompt_text, answer)
            
        answer = _clean_answer_output(answer)
        
        logs.extend([
            f"parse_answer_from_doc: Parsed answer: '{answer}'",
            f"parse_answer_from_doc: Document length: {len(document)} chars",
            f"parse_answer_from_doc: Prompt: {prompt_text}"
        ])
        
        return {
            "answer": answer,
            "logs": logs
        }
    except Exception as e:
        return {
            "answer": f"LLM parsing error: {str(e)}",
            "logs": [f"parse_answer_from_doc: Error: {str(e)}"]
        }

""" Below are rule-based step functions equivalent to the LLM-based ones above """
def infer_parameters_from_task_withrules(context: Dict[str, Any]) -> Dict[str, Any]:
    # Extract parameters from genomics task, rule-based. This is domain-specific logic for NCBI/genomics tasks.
    task = context.get("task", "")
    parameters = context.get("parameters", {})
    logs = []
    # Domain-specific parameter extraction for genomics
    result = {
        "database": "gene",  # Default database
        "search_term": "",
        "retmax": "5",
        "retmode": "json"
    }
    
    # Extract search term from common genomics question patterns
    if "official gene symbol" in task.lower():
        # Extract gene name from question like "What is the official gene symbol of SEP3?"
        match = re.search(r"gene symbol of ([A-Za-z0-9_-]+)", task)
        if match:
            result["search_term"] = match.group(1)
            result["database"] = "gene"
    
    elif "genes related to" in task.lower():
        # Disease-gene association questions
        match = re.search(r"genes related to (.+)\?", task)
        if match:
            result["search_term"] = match.group(1).strip()
            result["database"] = "omim"
    
    elif "chromosome" in task.lower() and "gene" in task.lower():
        # Gene location questions
        match = re.search(r"chromosome is ([A-Za-z0-9_-]+) gene", task)
        if match:
            result["search_term"] = match.group(1)
            result["database"] = "gene"
    
    elif "snp" in task.lower():
        # SNP-related questions
        if "associated with" in task.lower():
            match = re.search(r"SNP (rs\d+)", task)
            if match:
                result["search_term"] = match.group(1)
                result["database"] = "snp"
        elif "locate" in task.lower() or "chromosome" in task.lower():
            match = re.search(r"SNP (rs\d+)", task)
            if match:
                result["search_term"] = match.group(1)
                result["database"] = "snp"
    
    elif "convert" in task.lower() and "ensg" in task.lower():
        # Gene name conversion
        match = re.search(r"(ENSG\d+)", task)
        if match:
            result["search_term"] = match.group(1)
            result["database"] = "gene"
    
    elif "align" in task.lower() and "dna sequence" in task.lower():
        # DNA sequence alignment
        if "human genome" in task.lower():
            match = re.search(r"genome:([ATCG]+)", task)
            if match:
                result["search_term"] = match.group(1)
                result["database"] = "nt"
                result["tool"] = "blast"
        elif "organism" in task.lower() or "species" in task.lower():
            match = re.search(r"from:([ATCG]+)", task)
            if match:
                result["search_term"] = match.group(1)
                result["database"] = "nt"
                result["tool"] = "blast"
    
    elif "protein-coding gene" in task.lower():
        # Protein-coding gene questions
        match = re.search(r"Is ([A-Za-z0-9_-]+) a protein", task)
        if match:
            result["search_term"] = match.group(1)
            result["database"] = "gene"
    
    # Override with any explicitly provided parameters
    result.update(parameters)
    logs.append(f"infer_parameters_from_task_withrules: Extracted parameters: {result}")
    
    return {"parameters": result, "logs": logs}

def parse_blast_human_genome_alignment_withrules(context: Dict[str, Any]) -> Dict[str, Any]:
    """Parse BLAST results to get chromosome location format: chrX:start-end"""
    import re
    
    blast_result = context.get("document", "")
    logs = []
    
    if not blast_result:
        logs.append("parse_blast_human_genome_alignment_withrules: No BLAST document found in context")
        return {
            "answer": "No BLAST results to parse",
            "logs": logs
        }
    
    # This handles multi-line headers correctly
    sections = re.split(r'\n(?=>)', blast_result)
    
    # Filter out empty sections and the initial header
    alignment_sections = [s for s in sections if s.strip() and s.startswith('>')]
    
    logs.append(f"parse_blast_human_genome_alignment_withrules: {len(alignment_sections)} alignment sections found")
    
    for i, section in enumerate(alignment_sections):
        logs.append(f"parse_blast_human_genome_alignment_withrules: checking Section {i+1}")
        
        # This handles multi-line headers where chromosome info might be on any line
        header_lines = section.split('\n')[:5]  # Check first few lines of each section
        header_text = ' '.join(header_lines)
        
        logs.append(f"parse_blast_human_genome_alignment_withrules: Section {i+1} header: {header_text[:150]}...")
        
        # Check if it's a human chromosome (case-insensitive, handles various formats)
        chr_patterns = [
            r'chromosome\s+(\d+|[XY])',  # "chromosome 21"
            r'chr\s*(\d+|[XY])',         # "chr21" or "chr 21" 
            r'chromosome\s+(\d+|[XY])\b', # More specific boundary
        ]
        
        chromosome = None
        for pattern in chr_patterns:
            chr_match = re.search(pattern, header_text, re.IGNORECASE)
            if chr_match:
                chromosome = chr_match.group(1)
                break
        
        if not chromosome:
            logs.append(f"parse_blast_human_genome_alignment_withrules: Section {i+1}: No human chromosome found, skipping")
            continue
            
        logs.append(f"parse_blast_human_genome_alignment_withrules: Section {i+1}: Found chromosome {chromosome}")
        
                # Split section into individual alignments by looking for "Score =" patterns
        alignments = re.split(r'\n\s*Score\s*=', section)
        
        # Skip the header part (first element before any "Score =")
        if len(alignments) > 1:
            alignments = alignments[1:]  # Remove header part
            
            # Process alignments in order (first one is highest scoring)
            for j, alignment in enumerate(alignments):
                # Add back "Score =" that was removed by split
                alignment = "Score =" + alignment
                
                logs.append(f"parse_blast_human_genome_alignment_withrules: Section {i+1}, Alignment {j+1}")
                
                # Extract Sbjct positions from THIS specific alignment only
                sbjct_matches = re.findall(r'Sbjct\s+(\d+)\s+[ATCG\-\|\s]+\s+(\d+)', alignment)
                
                if sbjct_matches:
                    # Handle both forward and reverse strand alignments
                    all_positions = []
                    for match in sbjct_matches:
                        all_positions.extend([int(match[0]), int(match[1])])
                    
                    # Always use min and max to handle reverse strand correctly
                    start_pos = min(all_positions)
                    end_pos = max(all_positions)
                    
                    # Check if this is a reverse strand alignment
                    strand_match = re.search(r'Strand=(\w+)/(\w+)', alignment)
                    is_reverse = strand_match and strand_match.group(2) == "Minus"
                    
                    # VALIDATION: Calculate genomic span and compare with query length
                    genomic_span = end_pos - start_pos + 1
                    
                    # Extract query length from the alignment context
                    query_length = None
                    query_length_match = re.search(r'Length=(\d+)', blast_result)
                    if query_length_match:
                        query_length = int(query_length_match.group(1))
                    
                    answer = f"chr{chromosome}:{start_pos}-{end_pos}"
                    logs.append(f"parse_blast_human_genome_alignment_withrules: successfully parsed from first alignment: {answer}")
                    logs.append(f"parse_blast_human_genome_alignment_withrules: found {len(sbjct_matches)} Sbjct lines in this alignment")
                    logs.append(f"parse_blast_human_genome_alignment_withrules: strand orientation: {'reverse' if is_reverse else 'forward'}")
                    logs.append(f"parse_blast_human_genome_alignment_withrules: genomic span = {genomic_span} bp (max-min+1 = {end_pos}-{start_pos}+1)")
                    length_diff = 0
                    if query_length:
                        length_diff = genomic_span - query_length
                        logs.append(f"parse_blast_human_genome_alignment_withrules: query length = {query_length} bp")
                        logs.append(f"parse_blast_human_genome_alignment_withrules: validation check: genomic_span - query_length = {length_diff}")
                        if length_diff == 0:
                            logs.append(f"parse_blast_human_genome_alignment_withrules: ✓ VALIDATION PASSED: lengths match perfectly")
                        else:
                            logs.append(f"parse_blast_human_genome_alignment_withrules: ⚠ VALIDATION WARNING: length mismatch of {abs(length_diff)} bp")
                    else:
                        logs.append(f"parse_blast_human_genome_alignment_withrules: query length not found in document")
                    
                    return {
                        "answer": answer,
                        "logs": logs,
                        "validation": {
                            "genomic_span": genomic_span,
                            "query_length": query_length,
                            "length_difference": length_diff if query_length else None,
                            "validation_passed": length_diff == 0 if query_length else None,
                            "is_reverse_strand": is_reverse
                        }
                    }
                else:
                    logs.append(f"parse_blast_human_genome_alignment_withrules: Section {i+1}, Alignment {j+1}: No Sbjct positions found")
        else:
            logs.append(f"parse_blast_human_genome_alignment_withrules: Section {i+1}: No score-based alignments found")
    
    logs.append("parse_blast_human_genome_alignment_withrules: No suitable human chromosome alignment found")
    return {
        "answer": "No human chromosome alignment found",
        "logs": logs
    }

def parse_answer_from_doc_withrules(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse the final answer from the document based on the task type.
    """
    document = context.get("document", "")
    task = context.get("task", "")
    prompt = context.get("prompt", task)
    
    answer = "Unable to parse answer"
    
    try:
        # Gene alias questions
        if "official gene symbol" in task.lower():
            # Look for "Official Symbol: XXX" pattern
            match = re.search(r'Official Symbol:\s*([A-Za-z0-9_-]+)', document)
            if match:
                answer = match.group(1)
            else:
                # Fallback: look for gene symbol patterns
                lines = document.split('\n')
                for line in lines:
                    if 'Official Symbol:' in line:
                        parts = line.split('Official Symbol:')
                        if len(parts) > 1:
                            symbol = parts[1].split()[0].strip()
                            answer = symbol
                            break
        
        # Gene location questions
        elif "chromosome" in task.lower() and "gene" in task.lower():
            # Look for chromosome information
            match = re.search(r'Chromosome:\s*(\d+|X|Y)', document)
            if match:
                chr_num = match.group(1)
                answer = f"chr{chr_num}"
        
        # SNP association questions
        elif "snp" in task.lower() and "associated with" in task.lower():
            # Parse JSON response for SNP data
            try:
                if document.startswith('{'):
                    data = json.loads(document)
                    if "result" in data:
                        for uid, snp_data in data["result"].items():
                            if uid != "uids" and isinstance(snp_data, dict):
                                genes = snp_data.get("genes", [])
                                if genes and len(genes) > 0:
                                    answer = genes[0].get("name", "Unknown")
                                    break
            except json.JSONDecodeError:
                pass
        
        # Gene disease association
        elif "genes related to" in task.lower():
            # Extract gene symbols from OMIM results
            gene_symbols = []
            # Look for gene patterns in the text
            matches = re.findall(r'([A-Z][A-Z0-9_]+)(?:\s|,|$)', document)
            for match in matches:
                if len(match) >= 3 and match.isupper():
                    gene_symbols.append(match)
            
            if gene_symbols:
                # Remove duplicates and take first few
                unique_genes = list(dict.fromkeys(gene_symbols))
                answer = ", ".join(unique_genes[:5])  # Limit to 5 genes
        
        # Gene name conversion
        elif "convert" in task.lower() and "ensg" in task.lower():
            # Look for official symbol
            match = re.search(r'Official Symbol:\s*([A-Za-z0-9_-]+)', document)
            if match:
                answer = match.group(1)
        
        # Protein-coding gene questions
        elif "protein-coding gene" in task.lower():
            # Look for protein-coding indication
            if "protein-coding" in document.lower() or "codes for" in document.lower():
                answer = "TRUE"
            elif "pseudo" in document.lower() or "non-coding" in document.lower():
                answer = "NA"
            else:
                answer = "TRUE"  # Default assumption for genes
        
    except Exception as e:
        answer = f"Parse error: {str(e)}"
    
    return {
        "answer": answer,
        "reasoning": f"Parsed from document using pattern matching for task type",
        "metadata": {
            "document_length": len(document),
            "task_type": "genomics_query"
        }
    }

""" Below are helper functions used by the steps above """
# helper function to process API response based on return mode ('json', 'xml', 'text'), returning a dictionary with content and metadata.
def _process_response_by_mode(response_bytes: bytes, retmode: str) -> Dict[str, Any]:
    response_str = response_bytes.decode('utf-8')
    # Try to detect actual format regardless of requested retmode
    if response_str.strip().startswith('{') or response_str.strip().startswith('['):
        # JSON response
        try:
            parsed_json = json.loads(response_str)
            return {
                "content": parsed_json,
                "raw_response": response_str,
                "actual_format": "json",
                "requested_format": retmode
            }
        except json.JSONDecodeError:
            pass
    elif response_str.strip().startswith('<'):
        # XML response
        return {
            "content": response_str,
            "raw_response": response_str,
            "actual_format": "xml",
            "requested_format": retmode
        }
    # Default to text format
    return {
        "content": response_str,
        "raw_response": response_str,
        "actual_format": "text",
        "requested_format": retmode
    }

# helper function to load step-specific examples
def _load_step_examples(function_name: str, task_type: str, path_examples: str = PATH_EXAMPLES) -> List[Dict]:
    """Load few-shot examples for a specific function and task type."""
    try:
        config_path = Path(path_examples)
        if config_path.exists():
            with open(config_path, 'r') as f:
                data = json.load(f)
                examples = data.get(function_name, {}).get(task_type, [])
                return examples
        else:
            log_event(f"Warning: Examples file not found at {config_path}")
    except Exception as e:
        log_event(f"Warning: Could not load examples: {e}")
    
    return []

# clean common prefixes from LLM answer output.
def _clean_answer_output(answer: str) -> str:
    if not answer:
        return answer
    answer = answer.strip()
    # Common prefixes that LLMs might add (case-insensitive)
    prefixes_to_remove = [
        "Answer: ", "Answer:", "The answer is: ", "The answer is "
    ]
    for prefix in prefixes_to_remove:
        if answer.lower().startswith(prefix.lower()):
            return answer[len(prefix):].strip()
    return answer

# Build formatted examples text for few-shot prompting
def _build_examples_text(
    examples: List[Dict[str, Any]], 
    input_keys: List[str] = ['document', 'task'], 
    output_name: str = 'Answer'
) -> str:
    if not examples:
        return ""
    #examples_text = "\n\nHere are some examples:\n"
    examples_text = "\n\n[EXAMPLES - DO NOT USE THESE FOR YOUR ANSWER]\n"
    for i, ex in enumerate(examples, 1):  # Start numbering from 1
        examples_text += f"\n--- EXAMPLE {i} ---\n"
        # Handle input keys dynamically
        for key in input_keys:
            if key in ex["input"]:
                if key == 'document':
                    examples_text += f"Document: \n{ex['input'][key]}\n\n" # extra new line
                else:
                    examples_text += f"{key.title()}: {ex['input'][key]}\n"
        # Handle output (this is where JSON formatting happens if needed)
        if output_name.lower() == 'parameters':
            examples_text += f"{output_name}: {json.dumps(ex['output'])}\n"
        else:
            examples_text += f"{output_name}: {ex['output']}\n"
        
        examples_text += "---\n"
    
    examples_text += "[END OF EXAMPLES]\n" 
    return examples_text

# Build API context section for prompts
def _build_api_text(api_urls: List[str]) -> str:
    if not api_urls:
        return ""
    api_text = "\nThe document below is obtained using the following API calls:\n"
    for i, url in enumerate(api_urls, 1):
        # Extract key info from URL for readability
        if "esearch" in url:
            api_type = "Search"
        elif "efetch" in url:
            api_type = "Fetch"
        elif "esummary" in url:
            api_type = "Summary"
        elif "blast" in url:
            api_type = "BLAST"
        else:
            api_type = "API"
        
        api_text += f"{i}. {api_type}: {url}\n"
    
    api_text += "\n"
    return api_text

'''
Adjust MAX_DOCUMENT_SIZE based on appropriate rate limits:
Conservative: 30,000 chars (~7,500 tokens)
Moderate: 50,000 chars (~12,500 tokens)
Aggressive: 80,000 chars (~20,000 tokens)
'''
def _smart_truncate_blast_for_parsing(document, max_size=MAX_DOCUMENT_SIZE):
    """
    Smart truncation for BLAST documents - keeps the most important parts
    """
    if len(document) <= max_size:
        return document
    log_event(f"_smart_truncate_blast_for_parsing: doc max_size set as {max_size} chars")
    log_event(f"_smart_truncate_blast_for_parsing: doc length BEFORE is {len(document)}")
    lines = document.split('\n')
    kept_lines = []
    current_size = 0
    
    # Always keep the header and summary sections
    in_alignments = False
    alignment_count = 0
    max_alignments = 10  # Keep first 10 alignments
    
    for line in lines:
        # Check if we've hit our size limit
        if current_size + len(line) > max_size:
            kept_lines.append("[Document truncated to avoid rate limits]")
            break
            
        # Track alignment sections
        if line.startswith('>'):
            alignment_count += 1
            in_alignments = True
            
        # Skip alignments after the first few
        if in_alignments and alignment_count > max_alignments:
            if line.startswith('>'):
                kept_lines.append("[Additional alignments truncated]")
                break
        
        kept_lines.append(line)
        current_size += len(line) + 1  # +1 for newline
    
    doc_truncated = '\n'.join(kept_lines)
    log_event(f"_smart_truncate_blast_for_parsing: doc length BEFORE is {len(doc_truncated)} chars")
    return doc_truncated