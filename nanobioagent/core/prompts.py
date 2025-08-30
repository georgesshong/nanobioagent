"""
Prompt templates and examples for GeneGPT system.
Moved from main.py to centralize all prompt-related functionality.
"""

import re
import time
from ..tools.gene_utils import call_api, log_event

# Configuration constants
SYSTEM_PROMPT = 'You are a helpful assistant.\nWhen you have completed your answer, write "Answer: [your answer]" followed by two newlines (i.e. "\n\n") to indicate completion.\nFor accuracy, you should use the NCBI Web APIs, Eutils and BLAST (with examples in user prompt), to obtain the data.\n'
STOP_SEQUENCE = ['->', '\n\nQuestion']
MAX_NUM_CALLS = 10

# Prompt header generation functions
# v0: +"0", original geneGPT version
# v_: +"", current improved version
# v1: +"1" version 1

def get_prompt_header(mask):
    '''
    mask: [1/0 x 6], denotes whether each prompt component is used

    output: prompt
    '''
    url_1 = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=gene&retmax=5&retmode=json&sort=relevance&term=LMP10'
    call_1 = call_api(url_1)

    #url_2 = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=gene&retmax=5&retmode=json&id=19171,5699,8138'
    url_2 = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=gene&retmax=5&retmode=json&id=5699,8138,19171'
    call_2 = call_api(url_2)

    url_3 = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=snp&retmax=10&retmode=json&id=1217074595' 
    call_3 = call_api(url_3)

    url_4 = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=omim&retmax=20&retmode=json&sort=relevance&term=Meesmann+corneal+dystrophy'
    call_4 = call_api(url_4)

    #url_5 = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=omim&retmax=20&retmode=json&id=618767,601687,300778,148043,122100'
    url_5 = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=omim&retmax=20&retmode=json&id=122100,618767,620763,601687,148043'
    call_5 = call_api(url_5)

    url_6 = 'https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi?CMD=Put&PROGRAM=blastn&MEGABLAST=on&DATABASE=nt&FORMAT_TYPE=XML&QUERY=ATTCTGCCTTTAGTAATTTGATGACAGAGACTTCTTGGGAACCACAGCCAGGGAGCCACCCTTTACTCCACCAACAGGTGGCTTATATCCAATCTGAGAAAGAAAGAAAAAAAAAAAAGTATTTCTCT&HITLIST_SIZE=5'
    call_6 = call_api(url_6)
    # rid = re.search('RID = (.*)\n', call_6.decode('utf-8')).group(1)
    rid_match = re.search('RID = (.*)\n', call_6.decode('utf-8'))
    rid = ""
    if rid_match:
        rid = rid_match.group(1)
        

    url_7 = f'https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi?CMD=Get&FORMAT_TYPE=Text&RID={rid}'
    # time.sleep(30)
    call_7 = call_api(url_7)

    url_8 = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=nuccore&retmax=5&retmode=json&sort=relevance&term=RP11-255A11'
    call_8 = call_api(url_8)
    
    url_9 = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=nuccore&retmax=5&retmode=json&id=8574139'
    call_9 = call_api(url_9)

    url_10 = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=gene&retmax=5&retmode=json&sort=relevance&term=TTTY7'
    call_10 = call_api(url_10)
    
    url_11 = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=gene&retmax=5&retmode=json&id=246122'
    call_11 = call_api(url_11)

    url_12 = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=gene&retmax=5&retmode=json&sort=relevance&term=ENSG00000215251'
    call_12 = call_api(url_12)
    
    url_13 = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=gene&retmax=5&retmode=json&id=60493'
    call_13 = call_api(url_13)
    
    prompt = ''
    prompt += 'Hello. Your task is to use NCBI Web APIs to answer genomic questions.\n'
    #prompt += 'There are two types of Web APIs you can use: Eutils and BLAST.\n\n'
    prompt += 'For accuracy, you should use the two Web APIs: Eutils and BLAST, to obtain the data.\n\n'
    #prompt += 'Explain your workings and conclusion. And always end the conversation after that by repeating a final, succinct and to-the-point answer to the Question in the format of "Answer: [your response]".\n'
    prompt += 'When you have completed your answer, write "Answer: [your concise answer]" followed by two newlines (i.e. \n\n) to indicate completion.'
    prompt += 'When making an API call, you must enclose the URL in square brackets "[", "]" followed by "->" like "[URL]->" for it to be processed.\n'
    prompt += 'There will be maximum ' + str(MAX_NUM_CALLS) + ' API calls (indicated by number of "->") in the prompt, so you should work out an answer before that limit.\n'
 
    if mask[0]:
        # Doc 0 is about Eutils
        prompt += 'You can call Eutils by: "[https://eutils.ncbi.nlm.nih.gov/entrez/eutils/{esearch|efetch|esummary}.fcgi?db={gene|snp|omim}&retmax={}&{term|id}={term|id}]".\n'
        prompt += 'esearch: input is a search term and output is database id(s).\n'
        prompt += 'efectch/esummary: input is database id(s) and output is full records or summaries that contain name, chromosome location, and other information.\n'
        prompt += 'Normally, you need to first call esearch to get the database id(s) of the search term, and then call efectch/esummary to get the information with the database id(s).\n'
        prompt += 'For esearch calls, always include "&sort=relevance" in the query string behind "?".\n'
        prompt += 'For non-numerical id like ENSG (Ensembl Gene) such as "ENSG00000149476", always use "term=ENSG00000149476" and NOT "id=ENSG00000149476". \n'
        prompt += 'For numerical id or list of id separated by "," such as "idlist": ["5699"]  or "idlist": ["19171,5699,8138"] in the idlist, please use "id=5699" and "id=19171,5699,8138" respectively, not "term=".\n'
        prompt += 'If a non-numeric gene id contains a "." character in the string, you must change the id string to be the sub-string that comes before the "." character instead AND switch the db to be "nuccore" in the web api call. For example: change "RP11-255A11.4" to "RP11-255A11" (dropping ".4" and then use the query string of "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=nuccore&retmax=5&retmode=json&sort=relevance&term=RP11-255A11" instead. Do this ONLY if there is a "." dot character.\n'
        prompt += 'Database: gene is for genes, snp is for SNPs, and omim is for genetic diseases. nuccore is also for genes but only in the case when the id contains one "." followed by a number (the clone convention).\n\n'

    if mask[1]:
        # Doc 1 is about BLAST
        prompt += 'For DNA sequences, you can use BLAST by: "[https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi?CMD={Put|Get}&PROGRAM=blastn&MEGABLAST=on&DATABASE=nt&FORMAT_TYPE={XML|Text}&QUERY={sequence}&HITLIST_SIZE={max_hit_size}]".\n'
        prompt += 'BLAST maps a specific DNA {sequence} to its chromosome location among different specices.\n'
        prompt += 'You need to first PUT the BLAST request and then GET the results using the RID returned by PUT.\n\n'

    if any(mask[2:]):
        prompt += 'Here are some examples:\n\n'

    if mask[2]:
        # Example 1 is from gene alias task 
        prompt += f'Question: What is the official gene symbol of LMP10?\n'
        prompt += f'[{url_1}]->[{call_1}]\n' 
        prompt += f'[{url_2}]->[{call_2}]\n'
        prompt += f'Answer: PSMB10\n\n'
        
        prompt += f'Question: Convert ENSG00000215251 to official gene symbol.\n'
        prompt += f'[{url_12}]->[{call_12}]\n' 
        prompt += f'[{url_13}]->[{call_13}]\n'
        prompt += f'Answer: FASTKD5\n\n'

    if mask[3]:
        # Example 2 is from SNP gene task
        prompt += f'Question: Which gene is SNP rs1217074595 associated with?\n'
        prompt += f'[{url_3}]->[{call_3}]\n'
        prompt += f'Answer: LINC01270\n\n'

    if mask[4]:
        # Example 3 is from gene disease association
        prompt += f'Question: What are genes related to Meesmann corneal dystrophy?\n'
        prompt += f'[{url_4}]->[{call_4}]\n'
        prompt += f'[{url_5}]->[{call_5}]\n'
        prompt += f'Answer: KRT12, KRT3\n\n'
        # Example 3 is from gene location
        prompt += f'Question: Which chromosome is TTTY7 gene located on human genome?\n'
        prompt += f'[{url_10}]->[{call_10}]\n'
        prompt += f'[{url_11}]->[{call_11}]\n'
        prompt += f'Answer: Chromosome Y\n\n'
        # Example 3 is from gene location
        prompt += f'Question: Which chromosome is RP11-255A11.4 gene located on human genome?\n'
        prompt += f'[{url_8}]->[{call_8}]\n'
        prompt += f'[{url_9}]->[{call_9}]\n'
        prompt += f'Answer: Chromosome 9\n\n'

    if mask[5]:
        # Example 4 is for BLAST
        prompt += f'Question: Align the DNA sequence to the human genome:ATTCTGCCTTTAGTAATTTGATGACAGAGACTTCTTGGGAACCACAGCCAGGGAGCCACCCTTTACTCCACCAACAGGTGGCTTATATCCAATCTGAGAAAGAAAGAAAAAAAAAAAAGTATTTCTCT\n'
        prompt += f'[{url_6}]->[{rid}]\n'
        prompt += f'[{url_7}]->[{call_7}]\n'
        prompt += f'Answer: chr15:91950805-91950932\n\n'

    return prompt

def get_prompt_header_genegpt(mask):
	'''
	mask: [1/0 x 6], denotes whether each prompt component is used

	output: prompt
	'''
	url_1 = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=gene&retmax=5&retmode=json&sort=relevance&term=LMP10'
	call_1 = call_api(url_1)

	url_2 = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=gene&retmax=5&retmode=json&id=19171,5699,8138'
	call_2 = call_api(url_2)

	url_3 = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=snp&retmax=10&retmode=json&id=1217074595' 
	call_3 = call_api(url_3)

	url_4 = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=omim&retmax=20&retmode=json&sort=relevance&term=Meesmann+corneal+dystrophy'
	call_4 = call_api(url_4)

	url_5 = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=omim&retmax=20&retmode=json&id=618767,601687,300778,148043,122100'
	call_5 = call_api(url_5)

	url_6 = 'https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi?CMD=Put&PROGRAM=blastn&MEGABLAST=on&DATABASE=nt&FORMAT_TYPE=XML&QUERY=ATTCTGCCTTTAGTAATTTGATGACAGAGACTTCTTGGGAACCACAGCCAGGGAGCCACCCTTTACTCCACCAACAGGTGGCTTATATCCAATCTGAGAAAGAAAGAAAAAAAAAAAAGTATTTCTCT&HITLIST_SIZE=5'
	call_6 = call_api(url_6)
	rid = re.search('RID = (.*)\n', call_6.decode('utf-8')).group(1)

	url_7 = f'https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi?CMD=Get&FORMAT_TYPE=Text&RID={rid}'
	time.sleep(30)
	call_7 = call_api(url_7)

	prompt = ''
	prompt += 'Hello. Your task is to use NCBI Web APIs to answer genomic questions.\n'
	#prompt += 'There are two types of Web APIs you can use: Eutils and BLAST.\n\n'

	if mask[0]:
		# Doc 0 is about Eutils
		prompt += 'You can call Eutils by: "[https://eutils.ncbi.nlm.nih.gov/entrez/eutils/{esearch|efetch|esummary}.fcgi?db={gene|snp|omim}&retmax={}&{term|id}={term|id}]".\n'
		prompt += 'esearch: input is a search term and output is database id(s).\n'
		prompt += 'efectch/esummary: input is database id(s) and output is full records or summaries that contain name, chromosome location, and other information.\n'
		prompt += 'Normally, you need to first call esearch to get the database id(s) of the search term, and then call efectch/esummary to get the information with the database id(s).\n'
		prompt += 'Database: gene is for genes, snp is for SNPs, and omim is for genetic diseases.\n\n'

	if mask[1]:
		# Doc 1 is about BLAST
		prompt += 'For DNA sequences, you can use BLAST by: "[https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi?CMD={Put|Get}&PROGRAM=blastn&MEGABLAST=on&DATABASE=nt&FORMAT_TYPE={XML|Text}&QUERY={sequence}&HITLIST_SIZE={max_hit_size}]".\n'
		prompt += 'BLAST maps a specific DNA {sequence} to its chromosome location among different specices.\n'
		prompt += 'You need to first PUT the BLAST request and then GET the results using the RID returned by PUT.\n\n'

	if any(mask[2:]):
		prompt += 'Here are some examples:\n\n'

	if mask[2]:
		# Example 1 is from gene alias task 
		prompt += f'Question: What is the official gene symbol of LMP10?\n'
		prompt += f'[{url_1}]->[{call_1}]\n' 
		prompt += f'[{url_2}]->[{call_2}]\n'
		prompt += f'Answer: PSMB10\n\n'

	if mask[3]:
		# Example 2 is from SNP gene task
		prompt += f'Question: Which gene is SNP rs1217074595 associated with?\n'
		prompt += f'[{url_3}]->[{call_3}]\n'
		prompt += f'Answer: LINC01270\n\n'

	if mask[4]:
		# Example 3 is from gene disease association
		prompt += f'Question: What are genes related to Meesmann corneal dystrophy?\n'
		prompt += f'[{url_4}]->[{call_4}]\n'
		prompt += f'[{url_5}]->[{call_5}]\n'
		prompt += f'Answer: KRT12, KRT3\n\n'

	if mask[5]:
		# Example 4 is for BLAST
		prompt += f'Question: Align the DNA sequence to the human genome:ATTCTGCCTTTAGTAATTTGATGACAGAGACTTCTTGGGAACCACAGCCAGGGAGCCACCCTTTACTCCACCAACAGGTGGCTTATATCCAATCTGAGAAAGAAAGAAAAAAAAAAAAGTATTTCTCT\n'
		prompt += f'[{url_6}]->[{rid}]\n'
		prompt += f'[{url_7}]->[{call_7}]\n'
		prompt += f'Answer: chr15:91950805-91950932\n\n'

	return prompt

# Generate a direct prompt for LLM-only approach (no API calls).
def get_prompt_direct_simple(question):
    return f"You are a specialist for genomics information. {question}"

# Generate a direct prompt with examples for LLM-only approach.
def get_prompt_direct(question):
    direct_prompt = f"""You are a specialized assistant for genomics information.
        Please answer the following genomic question directly using your internal knowledge.
        CRITICAL INSTRUCTION: Do NOT use any external tools, agents, plugins, or web searches. 
        
        Important: Provide just the precise answer without explanation.
        For gene alias questions, answer only the official gene symbol.
        For gene location questions, answer only the chromosome (e.g., "chr7" or "chrY").
        For SNP location questions, answer only the chromosome.
        For DNA alignment questions, answer only the location (e.g., "chr15:91950805-91950932").
        For species alignment questions, answer only the organism name in lowercase (e.g., "human", "mouse", "rat", "worm", "zebrafish", "yeast").
        For gene-disease association questions, answer only the list of gene symbols separated by commas.
        For protein-coding genes questions, answer only "TRUE" or "NA".
        For gene name conversion questions, answer only the official gene symbol.
        
        [EXAMPLES - DO NOT USE THE CONTENT FOR YOUR ACTUAL ANSWER]:
        Question: What is the official gene symbol of LMP10?
        Answer: PSMB10
        
        Question: Which gene is SNP rs1217074595 associated with?
        Answer: LINC01270
        
        Question: What are genes related to Meesmann corneal dystrophy?
        Answer: KRT12, KRT3
        
        Question: Which chromosome is TTTY7 gene located on human genome?
        Answer: chrY
        
        Question: Which chromosome is RP11-255A11.4 gene located on human genome?
        Answer: chr9
        
        Question: Align the DNA sequence to the human genome:ATTCTGCCTTTAGTAATTTGATGACAGAGACTTCTTGGGAACCACAGCCAGGGAGCCACCCTTTACTCCACCAACAGGTGGCTTATATCCAATCTGAGAAAGAAAGAAAAAAAAAAAAGTATTTCTCT
        Answer: chr15:91950805-91950932
        
        Question: Convert ENSG00000215251 to official gene symbol.
        Answer: FASTKD5
        
        Question: Is ATP5F1EP2 a protein-coding gene?
        Answer: NA        
        [END OF EXAMPLES]
        
        Question: {question}
        Answer: """

    return direct_prompt

# Formatting utility functions
def normalize_chromosome(text):
    """Normalize chromosome text to standard format."""
    # Match patterns like "chromosome 13" or "chromosome   X"
    match = re.match(r'chromosome\s*(\w+)', text.lower())
    if match:
        chrom = match.group(1).upper()
        return f'chr{chrom}'
    else:
        return text  # if no match, return as-is

def extract_answer(text):
    """
    Extract the answer from the model's response.
    Looks for the text after "Answer:" or returns the full text if not found.
    """
    if "Answer:" in text:
        # Split by "Answer:" and take everything after it
        answer_part = text.split("Answer:", 1)[1].strip()
        answer_part = normalize_chromosome(answer_part)
        return answer_part
    else:
        # If no "Answer:" marker is found, return the whole text or the last paragraph
        return text.strip()
