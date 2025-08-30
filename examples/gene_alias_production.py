#!/usr/bin/env python3
"""
Production Gene Alias Generator - Generate 200 New Questions
Scales up the working approach to generate 200 unique gene alias questions
"""

import requests
import time
import json
import re
from typing import Dict, List, Tuple, Set
import random

class ProductionGeneAliasGenerator:
    def __init__(self, email: str = "test@example.com"):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.email = email
        self.rate_limit_delay = 0.4  # 2.5 requests per second to be safe
        self.existing_aliases: Set[str] = set()
        self.generated_questions: Dict[str, str] = {}
        
    def load_existing_data(self, filepath: str):
        """Load existing gene alias data to avoid duplicates"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            gene_alias_data = data.get("Gene alias", {})
            for question, answer in gene_alias_data.items():
                # Extract alias from question
                match = re.search(r'gene symbol of ([^?]+)\?', question)
                if match:
                    alias = match.group(1).strip()
                    self.existing_aliases.add(alias.lower())
            
            print(f"✅ Loaded {len(self.existing_aliases)} existing aliases to avoid duplicates")
            return data
        except Exception as e:
            print(f"⚠️ Could not load existing data: {e}")
            return {}
    
    def get_gene_aliases(self, gene_symbol: str) -> List[Tuple[str, str]]:
        """Get aliases for a specific gene symbol"""
        try:
            # Search for the gene
            search_params = {
                'db': 'gene',
                'term': f'{gene_symbol}[Gene Name] AND Homo sapiens[Organism]',
                'retmode': 'json',
                'email': self.email
            }
            
            search_response = requests.get(f"{self.base_url}esearch.fcgi", params=search_params)
            search_data = search_response.json()
            gene_ids = search_data.get('esearchresult', {}).get('idlist', [])
            
            if not gene_ids:
                return []
            
            time.sleep(self.rate_limit_delay)
            
            # Get gene summary with aliases
            summary_params = {
                'db': 'gene',
                'id': gene_ids[0],
                'retmode': 'json',
                'email': self.email
            }
            
            summary_response = requests.get(f"{self.base_url}esummary.fcgi", params=summary_params)
            summary_data = summary_response.json()
            
            if 'result' not in summary_data or gene_ids[0] not in summary_data['result']:
                return []
            
            gene_info = summary_data['result'][gene_ids[0]]
            official_symbol = gene_info.get('name', '')
            aliases = gene_info.get('otheraliases', '')
            
            if not aliases:
                return []
            
            # Parse and filter aliases
            alias_list = [alias.strip() for alias in aliases.split(',')]
            valid_aliases = []
            
            for alias in alias_list:
                if (alias and 
                    alias != official_symbol and 
                    len(alias) > 1 and
                    alias.lower() not in self.existing_aliases and
                    not alias.isdigit() and  # Skip pure numbers
                    len(alias) < 20):  # Skip very long aliases
                    
                    valid_aliases.append((alias, official_symbol))
                    self.existing_aliases.add(alias.lower())  # Prevent future duplicates
            
            return valid_aliases
            
        except Exception as e:
            print(f"Error processing {gene_symbol}: {e}")
            return []
    
    def search_genes_by_term(self, search_term: str, max_genes: int = 20) -> List[str]:
        """Search for gene symbols by a search term"""
        try:
            search_params = {
                'db': 'gene',
                'term': f'"{search_term}"[All Fields] AND Homo sapiens[Organism]',
                'retmax': max_genes,
                'retmode': 'json',
                'email': self.email
            }
            
            search_response = requests.get(f"{self.base_url}esearch.fcgi", params=search_params)
            search_data = search_response.json()
            gene_ids = search_data.get('esearchresult', {}).get('idlist', [])
            
            if not gene_ids:
                return []
            
            time.sleep(self.rate_limit_delay)
            
            # Get gene symbols from IDs
            summary_params = {
                'db': 'gene',
                'id': ','.join(gene_ids),
                'retmode': 'json',
                'email': self.email
            }
            
            summary_response = requests.get(f"{self.base_url}esummary.fcgi", params=summary_params)
            summary_data = summary_response.json()
            
            gene_symbols = []
            if 'result' in summary_data:
                for gene_id in gene_ids:
                    if gene_id in summary_data['result']:
                        gene_info = summary_data['result'][gene_id]
                        symbol = gene_info.get('name', '')
                        if symbol:
                            gene_symbols.append(symbol)
            
            return gene_symbols
            
        except Exception as e:
            print(f"Error searching for {search_term}: {e}")
            return []
    
    def generate_from_known_genes(self, target_count: int = 50) -> int:
        """Generate questions from known genes that likely have aliases"""
        print(f"🎯 Generating questions from known genes (target: {target_count})")
        
        # Extended list of genes known to have aliases
        known_genes = [
            # From existing dataset
            "PSMB10", "SLC38A6", "FCGR3A", "FNDC11", "EOLA2", "QSOX2", "OR10A2",
            "NUP50", "MLLT10", "MRPL57", "SYT4", "ZBTB6", "PTRH1", "ASIP", "LCE2A",
            "SEPTIN3", "TWIST2", "METTL24", "RNGTT", "RNF7", "F3", "SLC25A33",
            "PCLO", "OR56A1", "E2F1", "ZNF366", "ALOX15", "LYPD6B", "PRKCH",
            "ARHGEF26", "AGAP9", "IFT22", "GLB1", "COX7B", "PLK5", "ANKRD60",
            
            # Additional genes likely to have aliases
            "TP53", "BRCA1", "BRCA2", "EGFR", "MYC", "RAS", "AKT1", "PIK3CA",
            "PTEN", "MDM2", "CDKN2A", "RB1", "VHL", "APC", "MLH1", "MSH2",
            "ERBB2", "KIT", "PDGFRA", "FLT3", "ABL1", "BCR", "MLL", "ETV6",
            "RUNX1", "WT1", "NF1", "NF2", "TSC1", "TSC2", "KRAS", "NRAS",
            "BRAF", "RAF1", "MAP2K1", "MAPK1", "JUN", "FOS", "MYB", "ETS1"
        ]
        
        count = 0
        for gene in known_genes:
            if count >= target_count:
                break
                
            print(f"Processing: {gene}")
            aliases = self.get_gene_aliases(gene)
            
            for alias, official_symbol in aliases:
                question = f"What is the official gene symbol of {alias}?"
                self.generated_questions[question] = official_symbol
                count += 1
                print(f"  ✓ {alias} -> {official_symbol}")
                
                if count >= target_count:
                    break
            
            time.sleep(self.rate_limit_delay)
        
        print(f"Generated {count} questions from known genes")
        return count
    
    def generate_from_categories(self, target_count: int = 150) -> int:
        """Generate questions from functional categories"""
        print(f"🔍 Generating questions from functional categories (target: {target_count})")
        
        # Functional categories likely to have many aliases
        categories = [
            "kinase", "phosphatase", "receptor", "transcription factor",
            "zinc finger", "immunoglobulin", "olfactory receptor", "histone",
            "cytochrome", "oxidase", "reductase", "transferase", "hydrolase",
            "protease", "ligase", "synthetase", "channel", "transporter",
            "binding protein", "enzyme", "hormone", "cytokine", "chemokine",
            "integrin", "cadherin", "collagen", "keratin", "actin", "myosin",
            "tubulin", "ribosomal", "mitochondrial", "nuclear", "membrane"
        ]
        
        initial_count = len(self.generated_questions)
        target_per_category = max(1, target_count // len(categories))
        
        for category in categories:
            if len(self.generated_questions) - initial_count >= target_count:
                break
                
            print(f"  Searching category: {category}")
            gene_symbols = self.search_genes_by_term(category, max_genes=15)
            
            category_count = 0
            for gene_symbol in gene_symbols:
                if category_count >= target_per_category:
                    break
                    
                aliases = self.get_gene_aliases(gene_symbol)
                for alias, official_symbol in aliases:
                    question = f"What is the official gene symbol of {alias}?"
                    if question not in self.generated_questions:
                        self.generated_questions[question] = official_symbol
                        category_count += 1
                        print(f"    ✓ {alias} -> {official_symbol}")
                        
                        if category_count >= target_per_category:
                            break
                
                time.sleep(self.rate_limit_delay)
        
        generated_this_round = len(self.generated_questions) - initial_count
        print(f"Generated {generated_this_round} questions from categories")
        return generated_this_round
    
    def generate_all_questions(self, target_total: int = 200) -> Dict[str, str]:
        """Generate all questions using multiple strategies"""
        print(f"🚀 Starting production generation of {target_total} gene alias questions")
        print("=" * 70)
        
        # Strategy 1: Known genes (25% of target)
        known_target = min(50, target_total // 4)
        self.generate_from_known_genes(known_target)
        
        # Strategy 2: Functional categories (75% of target)
        remaining_target = target_total - len(self.generated_questions)
        if remaining_target > 0:
            self.generate_from_categories(remaining_target)
        
        print(f"\n📊 FINAL RESULTS:")
        print("=" * 40)
        print(f"Total questions generated: {len(self.generated_questions)}")
        print(f"Target was: {target_total}")
        
        return self.generated_questions

def main():
    """Main function to generate 200 gene alias questions"""
    print("🧬 Production Gene Alias Generator")
    print("=" * 50)
    
    # Initialize generator
    generator = ProductionGeneAliasGenerator(email="your_email@example.com")
    
    # Load existing data to avoid duplicates
    existing_data = generator.load_existing_data('geneturing_updated.json')
    
    # Generate new questions
    new_questions = generator.generate_all_questions(target_total=200)
    
    if len(new_questions) > 0:
        print(f"\n✅ SUCCESS! Generated {len(new_questions)} new questions")
        
        # Show sample questions
        print(f"\n📝 Sample questions:")
        sample_items = list(new_questions.items())[:10]
        for i, (question, answer) in enumerate(sample_items, 1):
            print(f"{i:2d}. {question}")
            print(f"    Answer: {answer}")
        
        # Update existing data
        if existing_data:
            existing_data['Gene alias'].update(new_questions)
            total_alias_questions = len(existing_data['Gene alias'])
        else:
            existing_data = {"Gene alias": new_questions}
            total_alias_questions = len(new_questions)
        
        # Save updated data
        output_filename = 'geneturing_updated_with_200_aliases.json'
        with open(output_filename, 'w') as f:
            json.dump(existing_data, f, indent=2)
        
        print(f"\n💾 Saved to '{output_filename}'")
        print(f"📈 Total gene alias questions: {total_alias_questions}")
        print(f"🆕 New questions added: {len(new_questions)}")
        
        # Also save just the new questions for review
        new_questions_only = {
            "New Gene Alias Questions": new_questions,
            "Generation Info": {
                "total_generated": len(new_questions),
                "generation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "method": "NCBI API (esearch + esummary)",
                "strategies": ["known_genes", "functional_categories"]
            }
        }
        
        with open('new_gene_alias_questions_200.json', 'w') as f:
            json.dump(new_questions_only, f, indent=2)
        
        print(f"📋 New questions only saved to 'new_gene_alias_questions_200.json'")
        print(f"\n🎉 Gene Alias category complete! Ready for next category.")
        
    else:
        print("\n❌ No questions generated. Check API access and try again.")

if __name__ == "__main__":
    main()
