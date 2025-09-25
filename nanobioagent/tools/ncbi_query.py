#!/usr/bin/env python
"""
Queries NCBI to find the gene informatics
https://www.ncbi.nlm.nih.gov/books/NBK25501/
db: {pubmed, pmc, gene, protein, nuccore, nucleotide, snp, omim, popset}
"""

import urllib.parse
import json
import os
import argparse
import re
from .gene_utils import call_api, log_event, resolve_single_path

import importlib
from typing import Dict, List, Tuple, Callable, Optional, Any, Union
from sentence_transformers import SentenceTransformer

DEFAULT_RETMAX = 5  # Default number of records to return
DEFAULT_RETMAX_RETRY_MULTIPLIER = 3

# class for the NCBI query engine
class NCBIQueryEngine:
    """
    Core engine for NCBI data queries, handles all API interactions
    """
    def esearch(self, term, db="gene", retmax=DEFAULT_RETMAX, retmode="json", only_human=False):
        """Search NCBI database for the given term"""
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db={db}&retmax={retmax}&retmode={retmode}&sort=relevance&term={term}"
        if only_human:
            # url += '[gene]+AND+Homo+sapiens[orgn]' # or "txid9606[Organism]"
            url += '+AND+Homo+sapiens[orgn]' # or "txid9606[Organism]"
        response = call_api(url)
        try:
            content = json.loads(response)
            if 'esearchresult' in content:
                result = content['esearchresult']
                return {
                    'count': result.get('count', '0'),
                    'retmax': result.get('retmax', '0'),
                    'ids': result.get('idlist', []),
                    'db': db,
                    'raw_response': response.decode('utf-8')
                }
            return {'error': 'Unknown response format', 'content': response.decode('utf-8')}
        except Exception as e:
            return {'error': str(e), 'content': response.decode('utf-8')}

    def efetch(self, ids, db="gene", retmax=DEFAULT_RETMAX, retmode="text"):
        """Fetch details from NCBI database for the given IDs"""
        if isinstance(ids, list):
            ids = ','.join(ids)
        
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db={db}&retmax={retmax}&retmode={retmode}&id={ids}"
        response = call_api(url)
        try:
            # For gene data, the response is text format
            content = response.decode('utf-8')
            return {
                'content': content,
                'db': db,
                'raw_response': content
            }
        except Exception as e:
            return {'error': str(e), 'content': response.decode('utf-8')}

    def esummary(self, ids, db="gene", retmax=DEFAULT_RETMAX, retmode="json"):
        """Fetch summary from NCBI database for the given IDs"""
        if isinstance(ids, list):
            ids = ','.join(ids)
        
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db={db}&retmax={retmax}&retmode={retmode}&id={ids}"
        response = call_api(url)
        try:
            if retmode == 'json':
                content = json.loads(response)
                if 'result' in content:
                    result = content['result']
                    # Extract UIDs from the result
                    uids = result.get('uids', [])
                    # Create a dictionary with uid-specific data
                    uid_data = {uid: result.get(uid, {}) for uid in uids}
                    
                    return {
                        'uids': uids,
                        'uid_data': uid_data,
                        'db': db,
                        'raw_response': response.decode('utf-8')
                    }
                return {'error': 'Unknown response format', 'content': response.decode('utf-8')}
            else:
                # For non-JSON responses
                content = response.decode('utf-8')
                return {
                    'content': content,
                    'db': db,
                    'raw_response': content
                }
        except Exception as e:
            return {'error': str(e), 'content': response.decode('utf-8')}
    
    def blast_put(self, sequence, database="nt", program="blastn", format_type="XML", hitlist_size=5):
        """Submit a BLAST search"""
        url = f"https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi?CMD=Put&PROGRAM={program}&MEGABLAST=on&DATABASE={database}&FORMAT_TYPE={format_type}&QUERY={sequence}&HITLIST_SIZE={hitlist_size}"
        
        # Use the call_api function from gene_utils
        response = call_api(url)
        
        try:
            # Extract RID
            rid_match = re.search('RID = (.*)\n', response.decode('utf-8'))
            if rid_match:
                return rid_match.group(1)
            return {"error": "Failed to extract RID from BLAST submission"}
        except Exception as e:
            return {"error": str(e)}

    def blast_get(self, rid, format_type="Text"):
        """Retrieve BLAST results"""
        url = f"https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi?CMD=Get&FORMAT_TYPE={format_type}&RID={rid}"
        
        # Use the call_api function from gene_utils
        response = call_api(url)
        
        return {
            "content": response.decode('utf-8'),
            "raw_response": response
        }
    
    def execute_query(self, search_term, parser, retmax=DEFAULT_RETMAX, only_human=False, allow_retry=True, check_other_aliases=True):
        """
        Execute a standard query flow: search, fetch, parse
        
        Parameters:
        - search_term: The term to search for
        - parser: Parser object to use for processing results
        - retmax: Maximum number of results to return
        - only_human: Whether to filter for human results only
        - allow_retry: Whether to allow retry with larger retmax when no matches found
        
        Returns:
        - Parsed results or error message and details
        """
        # Pre-process the query parameters
        input_params = preprocess_query_params(search_term)
        
        # Step 1: Search for the term (if not snp) or use the id directly (if snp)
        if input_params['db'] == 'snp': # handles find_snp_location
            ids = input_params['param_value']
        else:
            search_result = self.esearch(input_params['param_value'], db=input_params['db'], retmax=retmax, only_human=only_human)
            
            # Handle errors or no results
            if 'error' in search_result:
                return f"Error in search: {search_result['error']}", None
            
            ids = search_result.get('ids', [])
            if not ids:
                return f"No results found for {search_term} in {input_params['db']} database", None
            
        # Step 2: Fetch the details
        if input_params['db'] == 'snp': # handles find_snp_location
            # Use esummary for IDs
            fetch_result = self.esummary(ids, db=input_params['db'], retmax=retmax, retmode=input_params["retmode"])
        else:
            # Use efetch for terms
            fetch_result = self.efetch(ids, db=input_params['db'], retmax=retmax)
        
        # Handle errors
        if 'error' in fetch_result:
            return f"Error fetching details: {fetch_result['error']}", None
        
        # Step 3: Parse the results with the provided parser
        result, details = parser.parse(fetch_result, search_term=search_term, only_human=only_human, check_other_aliases=check_other_aliases)
        
        # Step 4: Check if we found no matches and should retry with a larger retmax
        # Looking at the parsers, they return messages starting with these prefixes when no matches are found
        no_result_indicators = [
            "Could not find", 
            "No gene symbol found", 
            "No human gene found",
            "No associated genes found",
            "SNP", # For SNP errors like "SNP rs123 not found in human genome"
            "Could not find chromosome"
        ]
        
        # Check if the result indicates no matches were found
        no_matches_found = (
            result is None or 
            (isinstance(result, str) and any(result.startswith(prefix) for prefix in no_result_indicators))
        )
        
        if no_matches_found and allow_retry:
            # Triple the retmax for the retry
            new_retmax = retmax * DEFAULT_RETMAX_RETRY_MULTIPLIER
            log_event(f"No matches found with retmax={retmax}, retrying with retmax={new_retmax}")
            
            # Call execute_query recursively with the new retmax but disable further retries
            return self.execute_query(
                search_term=search_term,
                parser=parser,
                retmax=new_retmax,
                only_human=only_human,
                allow_retry=False,  # Prevent infinite recursion
                check_other_aliases=check_other_aliases
            )
        
        # Return the results from either the first attempt or the retry
        return result, details
    
    def execute_disease_query(self, search_term, parser, retmax=10):
        """Execute a disease association query flow: search, summary, parse"""
        # Format the search term for URL
        formatted_search_term = urllib.parse.quote_plus(search_term)
        
        # Step 1: Search for the disease
        search_result = self.esearch(formatted_search_term, db="omim", retmax=retmax)
        
        # Handle errors or no results
        if 'error' in search_result:
            return f"Error in search: {search_result['error']}", None
        
        ids = search_result.get('ids', [])
        if not ids:
            return f"No results found for {search_term} in OMIM database", None
        
        # Step 2: Fetch the disease details
        summary_result = self.esummary(ids, db="omim", retmax=retmax)
        
        # Handle errors
        if 'error' in summary_result:
            return f"Error fetching disease details: {summary_result['error']}", None
        
        # Step 3: Parse the results with the provided parser
        return parser.parse(summary_result, search_term)
    
    def execute_blast_query(self, sequence, parser, database="nt", program="blastn"):
        """Execute a BLAST query flow: submit, retrieve, parse"""
        # Step 1: Submit BLAST search (PUT)
        rid = self.blast_put(sequence, database, program)
        
        # Handle errors
        if not rid or isinstance(rid, dict) and 'error' in rid:
            error_msg = rid.get('error', 'Unknown error') if isinstance(rid, dict) else 'Failed to obtain RID'
            return f"Error submitting BLAST search: {error_msg}", None
        
        # Step 2: Retrieve BLAST results (GET)
        blast_result = self.blast_get(rid)
        
        # Handle errors
        if 'error' in blast_result:
            return f"Error retrieving BLAST results: {blast_result['error']}", None
        
        # Step 3: Parse results with the appropriate parser
        return parser.parse(blast_result, sequence)

# various classes for parsing NCBI results
class GeneSymbolParser:
    """Parser for extracting official gene symbols"""
    def parse(self, fetch_result, search_term, only_human=False, check_other_aliases=True):
        """
        Parse the fetched result to find the official gene symbol
        
        Parameters:
        - fetch_result: The result fetched from the API
        - search_term: The term that was searched for
        - only_human: If True, only consider human genes
        - check_other_aliases: If True, only match entries where search_term appears in Other Aliases
                            or is the official symbol itself
        """
        content = fetch_result.get('content', '')
        
        # Parse the result to find the official symbol
        # Split into entries based on numbered pattern
        entries = []
        entry_pattern = re.compile(r'^\d+\.\s+\w+', re.MULTILINE)
        entry_positions = [m.start() for m in entry_pattern.finditer(content)]
        
        # Add the end of the string for the last entry
        entry_positions.append(len(content))
        
        # Extract each entry based on the positions
        for i in range(len(entry_positions) - 1):
            entry = content[entry_positions[i]:entry_positions[i+1]].strip()
            if entry:
                entries.append(entry)
        
        # Filter out entries that have been replaced
        valid_entries = []
        for entry in entries:
            if "This record was replaced with GeneID:" not in entry and "was replaced with GeneID:" not in entry:
                valid_entries.append(entry)
        
        # If we have valid entries, search them for the official symbol
        if valid_entries:
            for entry in valid_entries:
                # Check if this is a human entry if only_human is True
                is_human_entry = '[Homo sapiens' in entry or '(human)' in entry
                
                # Skip this entry if we only want human genes and this isn't one
                if only_human and not is_human_entry:
                    continue
                
                # Look for official symbol first
                match = re.search(r'Official Symbol:\s*(\S+)', entry)
                if not match:
                    continue  # Skip if no official symbol found
                
                symbol = match.group(1)
                
                # If check_other_aliases is True, check if search_term appears in Other Aliases section
                # OR if search_term is the official symbol itself
                if check_other_aliases:
                    # First check if search_term is the official symbol itself
                    if search_term == symbol:
                        return symbol, {
                            'fetch_result': content,
                            'entry': entry,
                            'match_type': 'official_symbol'
                        }
                    
                    # Otherwise, extract Other Aliases section
                    aliases_match = re.search(r'Other Aliases:\s*([^\n]+)', entry)
                    if not aliases_match:
                        continue  # Skip if Other Aliases section not found
                    
                    # Get the list of aliases
                    aliases_text = aliases_match.group(1)
                    aliases = [alias.strip() for alias in aliases_text.split(',')]
                    
                    ''' change to below for cases like PLK-5P = PLK-5 and ASV1 = ASV
                    # Check if search_term is in the aliases (case sensitive)
                    if search_term not in aliases:
                        continue  # Skip if search_term is not in aliases
                    '''
                    # Check if search_term is a prefix of any alias (case sensitive)
                    if not any(alias.startswith(search_term) for alias in aliases):
                        continue  # Skip if search_term is not in aliases
                    
                    return symbol, {
                        'fetch_result': content,
                        'entry': entry,
                        'match_type': 'alias'
                    }
                else:
                    # If we're not checking aliases, just return the official symbol
                    return symbol, {
                        'fetch_result': content,
                        'entry': entry
                    }
            
            # If we didn't find a match with the current filters, try again without the human filter
            # (but only if we were filtering for humans in the first place)
            if only_human and not check_other_aliases:
                for entry in valid_entries:
                    match = re.search(r'Official Symbol:\s*(\S+)', entry)
                    if match:
                        symbol = match.group(1)
                        return symbol, {
                            'fetch_result': content,
                            'entry': entry
                        }
        
        # If no valid entries found, check if there are replaced entries and extract their IDs
        replaced_ids = []
        for entry in entries:
            replaced_match = re.search(r'replaced with GeneID: (\d+)', entry)
            if replaced_match:
                replaced_id = replaced_match.group(1)
                replaced_ids.append(replaced_id)
        
        # If we got here, we couldn't find a valid official symbol
        error_message = "Could not find official gene symbol"
        if check_other_aliases:
            error_message += f" matching '{search_term}' (as symbol or alias)"
        if only_human:
            error_message += " in human genes"
        
        return f"{error_message} in the response", {
            'fetch_result': content
        }

class GeneSymbolParserSNP:
    """Parser for extracting gene symbols from SNP XML data"""
    def parse(self, fetch_result, search_term, only_human=False, check_other_aliases=False):
        """Parse the fetched result to find gene symbols associated with SNPs"""
        content = fetch_result.get('content', '')
        
        # Use regex to find all GENE_E elements and their contained NAME elements
        gene_matches = re.findall(r'<GENE_E>[\s\n]*<NAME>([^<]+)</NAME>', content, re.DOTALL)
        
        if gene_matches:
            # Return the first gene found (or could return all of them)
            gene_symbol = gene_matches[0]
            return gene_symbol, {
                'fetch_result': content,
                'genes_found': gene_matches
            }
        
        # If no gene elements found
        return "No gene symbol found in SNP data", {
            'fetch_result': content
        }
        
class GeneLocationParser:
    """Parser for extracting gene chromosome locations"""
    def parse(self, fetch_result, search_term, only_human=False, check_other_aliases=False):
        """Parse the fetched result to find the gene location"""
        content = fetch_result.get('content', '')
        db = fetch_result.get('db', 'gene')
        
        # For gene database, split the content into individual gene entries
        if db == 'gene':
            # Split the content into gene entries (each entry starts with a number and a period)
            entries = re.split(r'\n\d+\.\s+', content)
            # Remove the first split which is empty or contains header info
            if entries and not entries[0].strip().startswith('Official Symbol'):
                entries = entries[1:]
            
            # Process each entry
            for entry in entries:
                # Check if this is a human gene when only_human is True
                if only_human:
                    if not re.search(r'Homo sapiens|human', entry, re.IGNORECASE):
                        continue  # Skip non-human entries
                
                # Pattern to match "Chromosome: X; Location: Xp11.2" format
                chromosome_pattern = r"Chromosome:\s+([^;]+);"
                match = re.search(chromosome_pattern, entry)
                
                if match:
                    chromosome = match.group(1).strip()
                    # Format as "chrX"
                    if not chromosome.startswith('chr'):
                        chromosome = f"chr{chromosome}"
                    return chromosome, {
                        'fetch_result': content,
                        'entry': entry
                    }
            
            # If no entries matched after filtering for human (if requested)
            if only_human:
                return f"No human gene found for {search_term}", None
            
        elif db == 'nuccore':
            # For nuccore, we need to check for human entries if requested
            if only_human and not re.search(r"Homo sapiens|human", content, re.IGNORECASE):
                return f"No human sequence found for {search_term}", None
                
            # Pattern to match chromosome in nuccore format
            chromosome_pattern = r"subtype chromosome,\s+name\s+\"([^\"]+)\""
            match = re.search(chromosome_pattern, content)
            
            if match:
                chromosome = match.group(1).strip()
                # Format as "chrX"
                if not chromosome.startswith('chr'):
                    chromosome = f"chr{chromosome}"
                return chromosome, {
                    'fetch_result': content
                }
        
        elif db == 'snp':
            # For SNP
            if only_human and not re.search(r'<TAX_ID>9606</TAX_ID>', content):
                return f"SNP {search_term} not found in human genome", None
        
            # Extract chromosome info using regex
            match = re.search(r'<CHR>([^<]+)</CHR>', content)
            
            if match:
                chromosome = match.group(1).strip()
                # Format as "chrX"
                if not chromosome.startswith('chr'):
                    chromosome = f"chr{chromosome}"
                return chromosome, {
                    'fetch_result': content
                }
            
        # If we couldn't find the chromosome using the patterns above
        return f"Could not find chromosome information for {search_term}", None

class DiseaseAssociationParser:
    """Parser for extracting disease-gene associations"""
    def parse(self, summary_result, search_term):
        """Parse the summary result to find disease-gene associations"""
        # Get the UID data from the summary result
        uid_data = summary_result.get('uid_data', {})
        
        # Extract gene symbols from the summary result
        gene_symbols = []
        
        # Process each UID entry
        for uid, data in uid_data.items():
            # Check if this is a gene entry (oid starts with *)
            oid = data.get('oid', '')
            if oid.startswith('*') or oid.startswith('+'):
                # Extract gene symbol from title (after the last semicolon)
                title = data.get('title', '')
                if '; ' in title:
                    gene_symbol = title.split('; ')[-1]
                    gene_symbols.append(gene_symbol)
        
        if not gene_symbols:
            return f"No associated genes found for {search_term}", None
        
        # Convert the list to a comma-separated string
        gene_symbols_str = ", ".join(gene_symbols)
        
        return gene_symbols_str, {
            'summary_result': summary_result
        }

class BlastAlignmentParser:
    """Parser for extracting human genome DNA alignments from BLAST results"""
    
    def parse(self, blast_result, sequence):
        """Parse the BLAST result to find human genome alignments"""
        content = blast_result.get('content', '')
        
        if not content:
            return "No BLAST results to parse", None
        
        # Split into alignment sections using '>' as delimiter
        sections = re.split(r'\n(?=>)', content)
        alignment_sections = [s for s in sections if s.strip() and s.startswith('>')]
        
        # Process each alignment section
        for section in alignment_sections:
            # Handle multi-line headers - check first few lines
            header_lines = section.split('\n')[:5]
            header_text = ' '.join(header_lines)
            
            # Look for human chromosome with flexible patterns
            chr_patterns = [
                # Match "chromosome X" or "chromosome Xq21.2-22.2" - extract just the base chromosome
                r'chromosome\s+([XY](?:[pq][\d\.-]*)?|\d+(?:[pq][\d\.-]*)?|MT)',
                # Match "chr19" or "chr19:12345" - extract just the base chromosome  
                r'chr\s*([XY](?:[pq][\d\.-]*)?|\d+(?:[pq][\d\.-]*)?|MT)',
                # Match "from 2" or "from X" 
                r'from\s+([XY]|\d+|MT)(?:[,\s]|$)',
            ]
            
            chromosome = None
            for pattern in chr_patterns:
                chr_match = re.search(pattern, header_text, re.IGNORECASE)
                if chr_match:
                    # Extract just the base chromosome (X, Y, MT, or number)
                    full_match = chr_match.group(1)
                    # Extract just the chromosome letter/number part
                    base_chr_match = re.match(r'([XY]|\d+|MT)', full_match, re.IGNORECASE)
                    if base_chr_match:
                        chromosome = base_chr_match.group(1)
                        break
            
            if not chromosome:
                continue  # Skip this section, try next one
            
            # FIX: Extract positions from FIRST alignment only, not entire section
            # Split section into individual alignments by "Score =" 
            alignments = re.split(r'\n\s*Score\s*=', section)
            
            if len(alignments) > 1:
                # Take the first alignment (highest scoring)
                first_alignment = "Score =" + alignments[1]
                
                # Extract subject positions from FIRST alignment only
                pos_matches = re.findall(r'Sbjct\s+(\d+)\s+[ATCG\-\|\s]+\s+(\d+)', first_alignment)
                
                if pos_matches:
                    # Handle both forward and reverse strand alignments
                    all_positions = []
                    for match in pos_matches:
                        all_positions.extend([int(match[0]), int(match[1])])
                    
                    # Always use min and max to handle reverse strand correctly
                    start = min(all_positions)
                    end = max(all_positions)
                    
                    # Check if this is a reverse strand alignment
                    strand_match = re.search(r'Strand=(\w+)/(\w+)', first_alignment)
                    is_reverse = strand_match and strand_match.group(2) == "Minus"
                    
                    # VALIDATION: Calculate genomic span and compare with query length
                    genomic_span = end - start + 1
                    
                    # Extract query length from BLAST header
                    query_length = None
                    query_length_match = re.search(r'Length=(\d+)', content)
                    if query_length_match:
                        query_length = int(query_length_match.group(1))
                    
                    # Format the result as chrX:start-end
                    location = f"chr{chromosome}:{start}-{end}"
                    
                    # Create validation info
                    validation_info = {
                        'genomic_span': genomic_span,
                        'query_length': query_length,
                        'length_difference': genomic_span - query_length if query_length else None,
                        'validation_passed': (genomic_span == query_length) if query_length else None,
                        'is_reverse_strand': is_reverse
                    }
                    
                    return location, {
                        'blast_result': content,
                        'chromosome': chromosome,
                        'start': start,
                        'end': end,
                        'validation': validation_info
                    }
        
        return "Could not find human genome alignment for the sequence", None

class BlastOrganismParser:
    """Parser for identifying the organism from DNA sequence using BLAST"""
    def parse(self, blast_result, sequence):
        """Parse BLAST result to identify the organism of the DNA sequence"""
        content = blast_result.get('content', '')
        
        # Species mapping for GeneTuring
        species_map = {
            'Gallus gallus': 'chicken',
            'Gallus gallus (chicken)': 'chicken',
            'Cairina moschata': 'duck',
            'Cairina moschata breed yongchun': 'duck',
            'Homo sapiens': 'human',
            'Homo sapiens (human)': 'human',
            'human': 'human',
            'Mus musculus': 'mouse',
            'Mus musculus (house mouse)': 'mouse',
            'mouse': 'mouse',
            'Rattus norvegicus': 'rat',
            'Rattus norvegicus (rat)': 'rat',
            'rat': 'rat',
            'Meleagris gallopavo': 'turkey',
            'wild turkey': 'turkey',
            'Caenorhabditis elegans': 'worm',
            'Saccharomyces cerevisiae': 'yeast',
            "Saccharomyces cerevisiae (baker's yeast)": 'yeast',
            'Saccharomyces cerevisiae (yeast)': 'yeast',
            'yeast': 'yeast',
            'Danio rerio': 'zebrafish',
            'Zebrafish': 'zebrafish',
            'Zebrafish (Danio rerio)': 'zebrafish',
            'No significant similarity found': 'No significant similarity found'
        }
        # Look for the best hit
        for species in species_map.keys():
            if re.search(species, content):
                return species_map[species], {
                    'blast_result': content,
                    'species': species
                }
        
        # If no match in the mapping, try to extract the species name
        # species_pattern = re.search(r'>([A-Za-z ]+)', content)
        header_line_match = re.search(r'>(.*)\n', content)
        if header_line_match:
            header_line = header_line_match.group(1).strip()
            
            # Then try to extract just the species name
            species_pattern = re.search(r'>[^\s]+ ([A-Za-z]+ [A-Za-z]+)', content)
            if species_pattern:
                species_name = species_pattern.group(1).strip()
                return species_name, {
                    'blast_result': content,
                    'species': species_name,
                    'header_line': header_line,
                    'note': 'Species not in standard mapping'
                }
        
        # If all else fails, return the original error message
        return "Could not identify the organism from the DNA sequence", None

# helper functions
def looks_like_clone_symbol(symbol: str) -> bool:
    """
    Returns True if the input symbol looks like a clone-based or transcript-like gene symbol.
    """
    pattern = re.compile(r'^(RP\d+|CTD|CTB|C[HT]D|AC\d+|EN\w+)-?\w*\.\d+$', re.IGNORECASE)
    return bool(pattern.match(symbol))

def preprocess_query_params(input_value):
    """
    Prepare query parameters for NCBI API queries.
    Args:
        input_value (str): The input value to process
    Returns:
        dict: A dictionary containing:
            - param_value: The cleaned value for the query
            - param_type: The parameter type ('term' or 'id')
            - db: The database to query
            - retmode: The return mode ('json' or 'xml' or 'text')
    """
    # Initialize the result dictionary
    result = {
        "param_value": input_value,
        "param_type": "term",
        "db": "gene"  # Default database
    }
    
    # Process different input patterns
    if input_value.startswith("rs"):
        # SNP ID processing
        result["param_value"] = input_value.replace("rs", "")
        result["param_type"] = "id"
        result["db"] = "snp"
        result["retmode"] = "xml"
    
    elif input_value.startswith("ENSG"):
        # Ensembl gene ID processing
        result["param_type"] = "term"
        result["db"] = "gene"
    
    # elif "." in input_value:
    elif looks_like_clone_symbol(input_value):
        # Clean up by removing the trailing part after the last dot
        parts = input_value.split(".")
        result["param_value"] = ".".join(parts[:-1]) if len(parts) > 1 else input_value
        result["param_type"] = "term"
        result["db"] = "nuccore"
    
    elif all(nucleotide in "ATGC" for nucleotide in input_value.upper()):
        # DNA sequence processing
        result["param_type"] = "QUERY"
        result["db"] = "nt"
    
    return result


# Function wrappers to maintain the same interface
# 
# example: What is the official gene symbol of LMP10?
def find_official_symbol(search_term, retmax=DEFAULT_RETMAX, only_human=True, check_other_aliases=True):
    """Find the official gene symbol for a given search term"""
    engine = NCBIQueryEngine()
    parser = GeneSymbolParser()
    return engine.execute_query(search_term, parser, retmax=retmax, only_human=only_human, check_other_aliases=check_other_aliases)

# example: Which chromosome is FAM66D gene located on human genome?
def find_gene_location(search_term, retmax=DEFAULT_RETMAX, only_human=True):
    """Find the chromosome location of a gene"""
    engine = NCBIQueryEngine()
    parser = GeneLocationParser()
    return engine.execute_query(search_term, parser, retmax=retmax, only_human=only_human)

# example: What are genes related to Hemolytic anemia due to phosphofructokinase deficiency?
def find_disease_association(search_term, retmax=20, retmax_retry_multiplier=10):
    """Find genes associated with a disease"""
    engine = NCBIQueryEngine()
    parser = DiseaseAssociationParser()
    output = engine.execute_disease_query(search_term, parser, retmax=retmax)
    if isinstance(output[0], str) and output[0].startswith("No associated genes"):
        if retmax_retry_multiplier > 0:
            log_event(f"find_disease_association retrying with retmax={retmax * retmax_retry_multiplier}")
            output = engine.execute_disease_query(search_term, parser, retmax=retmax * retmax_retry_multiplier)
    return output

# example: Align the DNA sequence to the human genome:ATTCTGCCTTTAGTAATTTGATGACAGAGACTTCTTGGGAACCACAGCCAGGGAGCCACCCTTTACTCCACCAACAGGTGGCTTATATCCAATCTGAGAAAGAAAGAAAAAAAAAAAAGTATTTCTCT
def align_human_genome(sequence):
    """Align a DNA sequence to the human genome"""
    engine = NCBIQueryEngine()
    parser = BlastAlignmentParser()
    return engine.execute_blast_query(sequence, parser)

# example: Which organism does the DNA sequence come from:AGGGGCAGCAAACACCGGGACACACCCATTCGTGCACTAATCAGAAACTTTTTTTTCTCAAATAATTCAAACAATCAAAATTGGTTTTTTCGAGCAAGGTGGGAAATTTTTCGAT
def find_organism_from_dna(sequence):
    """Identify the organism from a DNA sequence"""
    engine = NCBIQueryEngine()
    parser = BlastOrganismParser()
    return engine.execute_blast_query(sequence, parser)

# example: Convert ENSG00000215251 to official gene symbol.
def convert_ensembl_to_official(ensembl_id, retmax=DEFAULT_RETMAX, only_human=True, check_other_aliases=False):
    return find_official_symbol(ensembl_id, retmax=retmax, only_human=only_human, check_other_aliases=check_other_aliases)

# example: Is ATP5F1EP2 a protein-coding gene?
def is_protein_coding(gene_name, retmax=DEFAULT_RETMAX):
    """Determine if a gene is protein-coding"""
    engine = NCBIQueryEngine()
    
    # Search for the gene
    search_result = engine.esearch(gene_name, db="gene", retmax=retmax)
    
    if 'error' in search_result or not search_result.get('ids'):
        return "NA", None
    
    # Fetch gene details
    fetch_result = engine.efetch(search_result['ids'], db="gene", retmax=retmax)
    
    if 'error' in fetch_result:
        return "NA", None
    
    content = fetch_result.get('content', '')
    
    return determine_protein_coding_status(content)

# Returns "TRUE" for protein-coding genes and "NA" for non-protein-coding genes
def determine_protein_coding_status(content):
    # Content is entirely lowercase for easier matching
    content_lower = content.lower()
    # Check for indicators of non-coding genes
    non_coding_indicators = [
        r'pseudo(?:gene)?',
        r'non[- ]coding',
        r'non[- ]protein',
        r'long non[- ]coding',
        r'lnc(?:rna)?',
        r'mir\d+',        # microRNA
        r'rnau\d+',       # RNA, U-class
        r'snor\d+',       # small nucleolar RNA
        r'trna',          # transfer RNA
        r'rrna',          # ribosomal RNA
        r'sirna',         # small interfering RNA
        r'antisense',     # antisense RNA
        r'ncrna'          # non-coding RNA
    ]
    
    for indicator in non_coding_indicators:
        if re.search(indicator, content_lower):
            return "NA", {'fetch_result': content, 'reason': f'Non-coding indicator found: {indicator}'}
    
    # Check for explicit indicators of protein-coding status
    if re.search(r'protein[- ]coding', content_lower):
        return "TRUE", {'fetch_result': content, 'reason': 'Explicit protein-coding mention'}
    
    # Check for protein types and functions (strong indicators of protein-coding)
    protein_indicators = [
        r'(?:growth|differentiation) factor',
        r'receptor',
        r'enzyme',
        r'kinase',
        r'channel',
        r'transporter',
        r'carrier',
        r'cytokine',
        r'chemokine',
        r'hormone',
        r'transcription factor',
        r'polymerase',
        r'protease',
        r'synthase',
        r'reductase',
        r'transferase',
        r'phosphatase',
        r'binding protein',
        r'domain',        # protein domain
        r'homolog',       # evolutionary homology suggests function
        r'inhibitor',     # Added for CAAP1-like cases
        r'activator',     # Added similar term
        r'regulator',     # Added similar term
        r'activity',      # Added for CAAP1 "caspase activity"
        r'apoptosis',     # Added for CAAP1-like cases
        r'signaling',     # Common protein function
        r'complex'        # Protein complex
    ]
    
    for indicator in protein_indicators:
        if re.search(indicator, content_lower):
            return "TRUE", {'fetch_result': content, 'reason': f'Protein function indicator found: {indicator}'}
    
    # NEW RULE ADDED FOR CASES LIKE CAAP1
    # Check for Other Designations that suggest protein function
    if re.search(r'Other Designations:', content) and not re.search(r'non-coding', content_lower):
        return "TRUE", {'fetch_result': content, 'reason': 'Has Other Designations section without non-coding indication'}
    
    # Check for MIM number - most genes with MIM (OMIM) entries are protein-coding
    if re.search(r'MIM:\s+\d+', content):
        return "TRUE", {'fetch_result': content, 'reason': 'Has MIM entry'}
    
    # ADDITIONAL NEW RULE FOR CASES LIKE CAAP1
    # If the gene has a standard human gene format with function description
    if re.search(r'Official Symbol:.+\[Homo sapiens \(human\)\]', content) and re.search(r'inhibitor|factor|protein|enzyme', content_lower):
        return "TRUE", {'fetch_result': content, 'reason': 'Standard human gene with functional description'}
    
    # If no clear indicators, default to "NA"
    return "NA", {'fetch_result': content, 'reason': 'No clear indicators found'}

# example: Which gene is SNP rs1217074595 associated with?
def find_snp_association(search_term, retmax=10):
    engine = NCBIQueryEngine()
    parser = GeneSymbolParserSNP()
    return engine.execute_query(search_term, parser, retmax)

# example: Which chromosome does SNP rs1430464868 locate on human genome?
def find_snp_location(snp_id, retmax=10, only_human=False):
    search_term = snp_id.replace("SNP ", "") if snp_id.startswith("SNP ") else snp_id
    return find_gene_location(search_term, retmax=retmax, only_human=only_human)

# class to map questions to the templates
class QuestionClassifier:
    def __init__(self, 
                mappings_file: str,
                module_name = None,
                model_name: str = "all-MiniLM-L6-v2",
                similarity_threshold: float = 0.7,
                use_patterns_first: bool = True):
        """
        Initialize the classifier with mappings from a JSON file.
        
        Args:
            mappings_file: Path to the JSON file containing mappings
            module_name: Name of the module containing the functions (optional)
            model_name: Name of the sentence transformer model to use
            similarity_threshold: Minimum similarity score to consider a match
            use_patterns_first: Whether to try pattern matching before using embeddings
        """
        # Load mappings from JSON
        with open(mappings_file, 'r') as f:
            mappings = json.load(f)
        
        self.template_questions = mappings.get("template_questions", {})
        self.pattern_maps = mappings.get("pattern_maps", {})
        function_mappings = mappings.get("function_mappings", {})
        
        # Convert function name strings to actual function references
        self.function_mapping = {}
        
        # This approach assumes the functions are defined in the current module
        # or in a specified module
        if module_name:
            module = importlib.import_module(module_name)
            for q_type, func_name in function_mappings.items():
                if hasattr(module, func_name):
                    self.function_mapping[q_type] = getattr(module, func_name)
        else:
            # Try to get functions from the global namespace
            globals_dict = globals()
            for q_type, func_name in function_mappings.items():
                if func_name in globals_dict and callable(globals_dict[func_name]):
                    self.function_mapping[q_type] = globals_dict[func_name]
        
        self.similarity_threshold = similarity_threshold
        self.use_patterns_first = use_patterns_first
        
        # Flatten templates for embedding
        self.flat_templates = []
        self.template_types = []
        
        for q_type, templates in self.template_questions.items():
            for template in templates:
                self.flat_templates.append(template)
                self.template_types.append(q_type)
        
        # After loading mappings
        # log_event(f"Loaded {len(self.flat_templates)} template questions")
        #for i, (template, q_type) in enumerate(zip(self.flat_templates, self.template_types)):
        #    log_event(f"  {i+1}: {q_type} - {template}")
    
        # Load the embedding model
        self.model = SentenceTransformer(model_name)
        self.template_embeddings = self.model.encode(self.flat_templates, convert_to_tensor=True)

    def classify_question(self, question: str, use_patterns_first: bool = True) -> Tuple[str, float]:
        """
        Determine the question type based on either direct pattern matching or embedding similarity.
        Args:
            question: The input question to classify
            use_patterns_first: If True, try pattern matching before using embeddings
        Returns:
            A tuple of (question_type, similarity_score)
        """
        # Try direct pattern matching first if enabled
        if use_patterns_first and hasattr(self, 'pattern_maps'):
            for q_type, patterns in self.pattern_maps.items():
                for pattern in patterns:
                    if re.search(pattern, question, re.IGNORECASE):
                        # Pattern match found - return with perfect score
                        # log_event(f"Pattern match found: {q_type} - {pattern}")
                        return q_type, 1.0
        
        # If no pattern match or patterns not enabled, fall back to embedding similarity
        import torch
        
        # Check if we have any templates
        if not self.flat_templates:
            return "unknown", 0.0
        
        # Encode the question
        question_embedding = self.model.encode(question, convert_to_tensor=True)
        
        # Make sure question_embedding is 2D for comparison
        if len(question_embedding.shape) == 1:
            question_embedding = question_embedding.unsqueeze(0)
        
        # Make sure template_embeddings is 2D
        if len(self.template_embeddings.shape) == 1:
            self.template_embeddings = self.template_embeddings.unsqueeze(0)
        
        # Check dimension alignment
        if question_embedding.shape[1] != self.template_embeddings.shape[1]:
            # Handle dimension mismatch
            log_event(f"Dimension mismatch: question {question_embedding.shape}, templates {self.template_embeddings.shape}")
            return "unknown", 0.0
        
        try:
            # Compute similarities with all templates
            similarities = torch.nn.functional.cosine_similarity(
                question_embedding, 
                self.template_embeddings, 
                dim=1
            )
            
            # Find the best match
            max_idx_tensor = torch.argmax(similarities)
            max_idx = int(max_idx_tensor.item())  # Explicit conversion to int
            max_score = float(similarities[max_idx].item())  # Explicit conversion to float
            
            if max_score >= self.similarity_threshold:
                return self.template_types[max_idx], max_score
            else:
                return "unknown", max_score
            
        except Exception as e:
            log_event(f"Error in similarity calculation: {str(e)}")
            return "unknown", 0.0
    
    def extract_argument(self, question: str, question_type: str) -> Any:
        """
        Extract the relevant argument(s) from the question based on its type
        Args:
            question: The input question
            question_type: The classified type of the question
        Returns:
            The extracted argument(s)
        """
        # Try each pattern for the given question type
        if question_type in self.pattern_maps:
            for pattern in self.pattern_maps[question_type]:
                match = re.search(pattern, question, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
        # If no pattern matches
        return None
    
    def process_question(self, question: str) -> Dict[str, Any]:
        """
        Complete pipeline for processing a question.
        Args:
            question: The input question
        Returns:
            Dictionary with results including type, argument, and answer
        """
        # Classify the question
        q_type, similarity = self.classify_question(question, self.use_patterns_first)
        
        if q_type == "unknown" or similarity < self.similarity_threshold:
            return {
                "question_type": "unknown",
                "similarity": similarity,
                "argument": None,
                "answer": None,
                "error": "Could not classify the question type"
            }
        
        # Extract the argument
        argument = self.extract_argument(question, q_type)
        
        if argument is None:
            return {
                "question_type": q_type,
                "similarity": similarity,
                "argument": None,
                "answer": None,
                "error": "Could not extract argument from the question"
            }
        
        # Get the appropriate function
        if q_type in self.function_mapping:
            function = self.function_mapping[q_type]
            
            # Call the function with the extracted argument
            try:
                answer = function(argument)
                return {
                    "question_type": q_type,
                    "similarity": similarity,
                    "argument": argument,
                    "answer": answer,
                    "error": None
                }
            except Exception as e:
                return {
                    "question_type": q_type,
                    "similarity": similarity,
                    "argument": argument,
                    "answer": None,
                    "error": f"Error calling function: {str(e)}"
                }
        else:
            return {
                "question_type": q_type,
                "similarity": similarity,
                "argument": argument,
                "answer": None,
                "error": f"No function mapping for question type: {q_type}"
            }

# overall wrapper function to handle gene queries
def gene_query(question: str, mappings_file: str = "data/question_mappings.json", module_name = None):
    """
    Main function to process a gene query question.
    Args:
        question: The input question to process
        mappings_file: Path to the JSON file containing mappings
        module_name: Name of the module containing the functions (optional)
    Returns:
        Dictionary with results including type, argument, and answer
    """
    mappings_file = resolve_single_path(mappings_file, "data/question_mappings.json")
        
    classifier = QuestionClassifier(mappings_file, module_name)
    result = classifier.process_question(question)
    # Check if there's an answer
    if result['answer'] is not None:
        # Handle both string answers and list answers
        if isinstance(result['answer'], list) and len(result['answer']) > 0:
            return result['answer'][0]
        else:
            return result['answer']
    else:
        # Return the error message or a default message
        if result['error']:
            return f"Error: {result['error']}"
        else:
            return "Sorry, I couldn't find an answer to this question."

# old main function, to replace
def main():
    parser = argparse.ArgumentParser(description='Find the official gene symbol')
    parser.add_argument('--term', required=True, help='Gene alias or name to search for')
    parser.add_argument('--output', required=True, help='Output file path')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Find the official gene symbol
    log_event(f"Looking up official gene symbol for: {args.term}")
    symbol, details = find_official_symbol(args.term)
    
    # Create the results dictionary
    results = {
        'query': args.term,
        'official_symbol': symbol,
        'details': details
    }
    
    # Save the results to the output file
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    log_event(f"Results saved to {args.output}")
    
    # Also print the answer for easy visibility in console
    log_event(f"Answer: {symbol}")

if __name__ == "__main__":
    main()