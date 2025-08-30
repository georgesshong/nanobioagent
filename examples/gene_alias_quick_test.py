#!/usr/bin/env python3
"""
Working Gene Alias Generator using NCBI API
Fixed version with proper API calls and parsing
"""

import requests
import time
import json
import re
import xml.etree.ElementTree as ET

def test_single_gene(gene_symbol, email="test@example.com"):
    """Test fetching aliases for a single known gene"""
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    
    print(f"Testing gene: {gene_symbol}")
    
    try:
        # Step 1: Search for the specific gene
        search_url = f"{base_url}esearch.fcgi"
        search_params = {
            'db': 'gene',
            'term': f'{gene_symbol}[Gene Name] AND Homo sapiens[Organism]',
            'retmode': 'json',
            'email': email
        }
        
        search_response = requests.get(search_url, params=search_params)
        search_data = search_response.json()
        gene_ids = search_data.get('esearchresult', {}).get('idlist', [])
        
        if not gene_ids:
            print(f"No gene ID found for {gene_symbol}")
            return []
        
        gene_id = gene_ids[0]
        print(f"Found gene ID: {gene_id}")
        
        time.sleep(0.4)  # Rate limiting
        
        # Step 2: Get gene summary which includes aliases
        summary_url = f"{base_url}esummary.fcgi"
        summary_params = {
            'db': 'gene',
            'id': gene_id,
            'retmode': 'json',
            'email': email
        }
        
        summary_response = requests.get(summary_url, params=summary_params)
        summary_data = summary_response.json()
        
        if 'result' not in summary_data or gene_id not in summary_data['result']:
            print("No summary data found")
            return []
        
        gene_info = summary_data['result'][gene_id]
        
        # Extract information
        official_symbol = gene_info.get('name', '')
        aliases = gene_info.get('otheraliases', '')
        description = gene_info.get('description', '')
        
        print(f"Official Symbol: {official_symbol}")
        print(f"Aliases: {aliases}")
        print(f"Description: {description}")
        
        # Parse aliases
        if aliases:
            alias_list = [alias.strip() for alias in aliases.split(',')]
            # Filter out empty and identical aliases
            valid_aliases = [
                alias for alias in alias_list 
                if alias and alias != official_symbol and len(alias) > 1
            ]
            return [(alias, official_symbol) for alias in valid_aliases]
        
        return []
        
    except Exception as e:
        print(f"Error processing {gene_symbol}: {e}")
        return []

def generate_from_known_genes():
    """Generate questions from known genes that have aliases"""
    
    # Known genes from the existing dataset that should have aliases
    known_genes = [
        "PSMB10",  # LMP10
        "SLC38A6", # SNAT6  
        "FCGR3A",  # IMD20
        "TWIST2",  # FFDD3
        "ALOX15",  # 15-LOX
        "PRKCH",   # PKCL
        "ARHGEF26", # SGEF
        "PLK5",    # PLK-5
        "SEPTIN3", # SEP3
        "ZBTB6",   # ZNF482
        "OR10A2",  # OR11-86
        "NUP50",   # NPAP60L
        "MLLT10",  # AF10
        "MRPL57",  # bMRP63
        "SYT4",    # HsT1192
        "E2F1",    # RBAP1
        "ZNF366",  # DCSCRIPT
        "LYPD6B",  # CT116
        "AGAP9",   # CTGLF6
        "IFT22",   # FAP9
        "GLB1"     # MPS4B
    ]
    
    generated_questions = {}
    
    for gene in known_genes:
        print(f"\n{'='*50}")
        alias_pairs = test_single_gene(gene)
        
        for alias, official_symbol in alias_pairs:
            question = f"What is the official gene symbol of {alias}?"
            generated_questions[question] = official_symbol
            print(f"Generated: {question} -> {official_symbol}")
        
        time.sleep(0.4)  # Rate limiting
        
        if len(generated_questions) >= 30:  # Stop at 30 for testing
            break
    
    return generated_questions

def search_genes_by_category():
    """Search for genes by functional categories"""
    
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    email = "test@example.com"
    
    # Use more specific search terms
    search_terms = [
        "protein kinase",
        "zinc finger", 
        "immunoglobulin",
        "olfactory receptor",
        "histone"
    ]
    
    generated_questions = {}
    
    for term in search_terms:
        print(f"\nSearching for: {term}")
        
        try:
            # Search for genes
            search_url = f"{base_url}esearch.fcgi"
            search_params = {
                'db': 'gene',
                'term': f'"{term}"[All Fields] AND Homo sapiens[Organism]',
                'retmax': 10,
                'retmode': 'json',
                'email': email
            }
            
            search_response = requests.get(search_url, params=search_params)
            search_data = search_response.json()
            gene_ids = search_data.get('esearchresult', {}).get('idlist', [])
            
            print(f"Found {len(gene_ids)} genes")
            
            if not gene_ids:
                continue
            
            time.sleep(0.4)
            
            # Get summaries for the genes
            summary_url = f"{base_url}esummary.fcgi"
            summary_params = {
                'db': 'gene',
                'id': ','.join(gene_ids[:5]),  # Take first 5
                'retmode': 'json',
                'email': email
            }
            
            summary_response = requests.get(summary_url, params=summary_params)
            summary_data = summary_response.json()
            
            if 'result' not in summary_data:
                continue
            
            for gene_id in gene_ids[:5]:
                if gene_id in summary_data['result']:
                    gene_info = summary_data['result'][gene_id]
                    official_symbol = gene_info.get('name', '')
                    aliases = gene_info.get('otheraliases', '')
                    
                    if aliases and official_symbol:
                        alias_list = [alias.strip() for alias in aliases.split(',')]
                        for alias in alias_list[:2]:  # Take first 2 aliases
                            if alias and alias != official_symbol and len(alias) > 1:
                                question = f"What is the official gene symbol of {alias}?"
                                generated_questions[question] = official_symbol
                                print(f"Generated: {question} -> {official_symbol}")
            
            time.sleep(0.4)
            
        except Exception as e:
            print(f"Error with {term}: {e}")
            continue
    
    return generated_questions

def main():
    print("Testing Gene Alias Generation with Fixed NCBI API")
    print("=" * 60)
    
    # Method 1: Test with known genes
    print("\n🧪 Method 1: Testing with known genes from existing dataset")
    known_gene_questions = generate_from_known_genes()
    
    print(f"\nGenerated {len(known_gene_questions)} questions from known genes")
    
    # Method 2: Search by categories
    print("\n🔍 Method 2: Searching by functional categories")
    category_questions = search_genes_by_category()
    
    print(f"\nGenerated {len(category_questions)} questions from category search")
    
    # Combine results
    all_questions = {**known_gene_questions, **category_questions}
    
    print(f"\n📊 RESULTS:")
    print("=" * 40)
    print(f"Total questions generated: {len(all_questions)}")
    
    if all_questions:
        print("\n✅ SUCCESS! Sample questions:")
        for i, (question, answer) in enumerate(list(all_questions.items())[:10]):
            print(f"{i+1:2d}. {question}")
            print(f"    Answer: {answer}")
        
        # Save results
        output_data = {
            "Generated Gene Alias Questions": all_questions,
            "Generation Info": {
                "method": "NCBI API (esearch + esummary)",
                "total_generated": len(all_questions),
                "from_known_genes": len(known_gene_questions),
                "from_category_search": len(category_questions)
            }
        }
        
        with open('working_gene_alias_test.json', 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n💾 Saved results to 'working_gene_alias_test.json'")
        print("🚀 Ready to scale up to 200 questions!")
        
    else:
        print("\n❌ No questions generated. Need to debug further.")
        print("\n🔧 Debugging tips:")
        print("1. Check internet connection")
        print("2. Verify NCBI API is accessible")
        print("3. Try different search terms")

if __name__ == "__main__":
    main()