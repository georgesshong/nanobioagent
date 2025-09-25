#!/usr/bin/env python3
"""
Model Configuration JSON to CSV Converter

Converts model configuration CSV files to JSON format and vice-versa with support for:
- Flattening nested pricing structures
- Including _defaults as first row
- Handling missing fields gracefully
- Command line usage

Usage:
    python convert_model_config.py input_file.json
    python convert_model_config.py input_file.csv
"""
# HARDCODED NVIDIA NIM PREFIXES - automatically added to JSON output
# add more prefixes as needed
NVIDIA_NIM_PREFIXES = [
    "deepseek-ai/", "google/", "ibm/", "llama-3.1-nemotron", "llama-3.3-nemotron",
    "nvidia/", "marin/", "meta/", "microsoft/", "mistralai/", "nv-mistralai/", 
    "qwen/", "tiiuae/", "zyphra/", "openai/"
]

import json
import csv
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Union

# true if a value represents a number, handling Excel formatting.
def is_numeric_value(value: Any) -> bool:
    if isinstance(value, (int, float)):
        return True
    if not value or value == "":
        return False
    try:
        # Clean up various apostrophe/quote characters used as thousands separators
        clean_value = str(value)
        # Remove various types of apostrophes and quotes
        for separator in ["’", "'", "'", "'", "`", ","]:
            clean_value = clean_value.replace(separator, "")
        clean_value = clean_value.strip()
        
        if not clean_value:
            return False
        # Try to convert to float to test if it's numeric
        float(clean_value)
        return True
    except (ValueError, TypeError):
        return False

# Parse a numeric value, handling Excel formatting and scientific notation.
def parse_numeric_value(value: Any) -> Union[int, float]:
    # Always returns integers when possible, avoiding scientific notation.
    if isinstance(value, (int, float)):
        return int(value) if isinstance(value, float) and value.is_integer() else value
    # Clean up various apostrophe/quote characters used as thousands separators
    clean_value = str(value)
    for separator in ["’", "'", "'", "'", "`", ","]:
        clean_value = clean_value.replace(separator, "")
    clean_value = clean_value.strip()
    
    if "E" in clean_value.upper() or "e" in clean_value:
        # Handle scientific notation - always convert to int to avoid E+ format
        return int(float(clean_value))
    elif '.' in clean_value:
        float_val = float(clean_value)
        # Return int if it's a whole number, otherwise float
        return int(float_val) if float_val.is_integer() else float_val
    else:
        return int(clean_value)

# flatten a model configuration dictionary, handling nested pricing structure.
def flatten_model_config(model_data: Dict[str, Any]) -> Dict[str, Any]:
    # takes in model_data: Dictionary containing model configuration
    # returns: Flattened dictionary with pricing.* keys
    flattened = {}
    
    for key, value in model_data.items():
        if key == "pricing" and isinstance(value, dict):
            # Flatten pricing structure
            for pricing_key, pricing_value in value.items():
                flattened[f"pricing.{pricing_key}"] = pricing_value
        else:
            flattened[key] = value
    
    return flattened


def get_all_columns(models_data: Dict[str, Any], defaults_data: Dict[str, Any]) -> List[str]:
    """
    Extract all possible column names from models and defaults data.
    Uses _defaults order as the primary template, then adds any additional columns.
    Args:
        models_data: Dictionary of all models
        defaults_data: Dictionary of default values
    Returns:
        List of column names ordered by _defaults structure, then additional columns
    """
    # Start with model_name
    columns_list = ["model_name"]
    
    # Get columns from defaults in their natural order (preserves JSON key order)
    flattened_defaults = flatten_model_config(defaults_data)
    defaults_columns = list(flattened_defaults.keys())
    columns_list.extend(defaults_columns)
    
    # Collect all additional columns from models that aren't in defaults
    additional_columns = set()
    for model_name, model_config in models_data.items():
        flattened_model = flatten_model_config(model_config)
        for col in flattened_model.keys():
            if col not in columns_list:
                additional_columns.add(col)
    
    # Add additional columns in alphabetical order at the end
    columns_list.extend(sorted(additional_columns))
    
    return columns_list

# create a CSV row for a model configuration.
def create_csv_row(model_name: str, model_data: Dict[str, Any], columns: List[str]) -> List[str]:
    """
    Args:
        model_name: Name of the model
        model_data: Model configuration dictionary
        columns: List of all column names    
    Returns:
        List of values corresponding to the columns
    """
    flattened = flatten_model_config(model_data)
    flattened["model_name"] = model_name
    
    row = []
    for column in columns:
        value = flattened.get(column, "")
        # Convert value to string, handling various types
        if value is None:
            row.append("")
        elif isinstance(value, (list, dict)):
            # Convert complex types to JSON strings
            row.append(json.dumps(value))
        else:
            row.append(str(value))
    
    return row


def normalize_line_endings(file_path: str, line_ending: str = '\n'):
    """
    Normalize line endings in a file.
    
    Args:
        file_path: Path to the file to normalize
        line_ending: Desired line ending ('\n' for LF, '\r\n' for CRLF)
    """
    with open(file_path, 'rb') as f:
        content = f.read()
    
    # Convert to string and normalize
    text = content.decode('utf-8')
    text = text.replace('\r\n', '\n').replace('\r', '\n')  # Normalize to LF
    
    if line_ending == '\r\n':
        text = text.replace('\n', '\r\n')  # Convert to CRLF if requested
    
    # Write back
    with open(file_path, 'wb') as f:
        f.write(text.encode('utf-8'))


def json_to_csv(input_file: str) -> str:
    """
    Convert model configuration JSON to CSV format.
    
    Args:
        input_file: Path to input JSON file
        
    Returns:
        Path to output CSV file
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        json.JSONDecodeError: If JSON is invalid
        KeyError: If required keys are missing
    """
    # Validate input file
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Generate output filename
    input_path = Path(input_file)
    output_file = str(input_path.with_suffix('.csv'))
    
    # Load JSON data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Validate required keys
    if "models" not in data:
        raise KeyError("JSON file must contain 'models' key")
    
    if "_defaults" not in data:
        raise KeyError("JSON file must contain '_defaults' key")
    
    models_data = data["models"]
    defaults_data = data["_defaults"]
    
    # Get all possible columns
    columns = get_all_columns(models_data, defaults_data)
    
    # Write CSV file with consistent line endings
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(columns)
        
        # Write defaults row first (with model_name as "_defaults")
        defaults_row = create_csv_row("_defaults", defaults_data, columns)
        writer.writerow(defaults_row)
        
        # Write model rows
        for model_name, model_config in models_data.items():
            model_row = create_csv_row(model_name, model_config, columns)
            writer.writerow(model_row)
    
    return output_file


def unflatten_model_config(flattened_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unflatten a model configuration dictionary, reconstructing nested pricing structure.
    
    Args:
        flattened_data: Flattened dictionary with pricing.* keys
        
    Returns:
        Dictionary with nested pricing structure
    """
    unflattened = {}
    pricing = {}
    
    for key, value in flattened_data.items():
        if key.startswith("pricing."):
            # Reconstruct pricing structure
            pricing_key = key.replace("pricing.", "")
            if value != "":  # Only add non-empty values
                # Try to convert to appropriate type
                try:
                    if is_numeric_value(value):
                        pricing[pricing_key] = parse_numeric_value(value)
                    else:
                        pricing[pricing_key] = value
                except (ValueError, TypeError):
                    pricing[pricing_key] = value
        elif key != "model_name" and value != "":
            # Handle other fields (skip empty values)
            try:
                # Try to parse JSON strings back to objects
                if value.startswith(('[', '{')):
                    unflattened[key] = json.loads(value)
                elif value.lower() in ('true', 'false'):
                    unflattened[key] = value.lower() == 'true'
                elif is_numeric_value(value):
                    unflattened[key] = parse_numeric_value(value)
                else:
                    unflattened[key] = value
            except (json.JSONDecodeError, ValueError):
                unflattened[key] = value
    
    if pricing:
        unflattened["pricing"] = pricing
    
    return unflattened


def clean_column_name(column_name: str) -> str:
    """
    Clean column names by removing various BOMs and whitespace.    
    Args:
        column_name: Raw column name from CSV
    Returns:
        Cleaned column name
    """
    if not column_name:
        return ""
    
    # Remove various BOM types
    boms_to_remove = [
        '\ufeff',      # UTF-8 BOM
        '\ufffe',      # UTF-16 BE BOM  
        '\xff\xfe',    # UTF-16 LE BOM
        '\x00\x00\xfe\xff',  # UTF-32 BE BOM
        '\xff\xfe\x00\x00',  # UTF-32 LE BOM
        '\xef\xbb\xbf',      # UTF-8 BOM as bytes
    ]
    
    cleaned = column_name
    for bom in boms_to_remove:
        cleaned = cleaned.replace(bom, '')
    
    # Remove leading/trailing whitespace and normalize internal whitespace
    cleaned = ' '.join(cleaned.split())
    
    return cleaned


def detect_encoding(file_path: str) -> str:
    """
    Detect the encoding of a file by trying common encodings.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Detected encoding name
    """
    encodings_to_try = ['utf-8', 'utf-8-sig', 'cp1252', 'latin-1', 'iso-8859-1']
    
    for encoding in encodings_to_try:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                f.read()
            return encoding
        except UnicodeDecodeError:
            continue
    
    # If all fail, use latin-1 as it can decode any byte sequence
    return 'latin-1'


def csv_to_json(input_file: str, preserve_metadata: bool = True) -> str:
    """
    Convert CSV back to model configuration JSON format.
    
    Args:
        input_file: Path to input CSV file
        preserve_metadata: Whether to preserve original metadata and other sections
        
    Returns:
        Path to output JSON file
        
    Raises:
        FileNotFoundError: If input file doesn't exist
    """
    # Validate input file
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Generate output filename
    input_path = Path(input_file)
    output_file = str(input_path.with_suffix('.json'))
    
    # Detect encoding of the CSV file
    detected_encoding = detect_encoding(input_file)
    print(f"🔍 Detected encoding: {detected_encoding}")
    
    # Read CSV data with detected encoding
    with open(input_file, 'r', encoding=detected_encoding) as csvfile:
        reader = csv.DictReader(csvfile)
        
        # Read all rows and clean up column names (strip whitespace)
        rows = []
        cleaned_fieldnames = None
        
        for row in reader:
            if cleaned_fieldnames is None:
                # Clean up fieldnames on first row - handle all BOM types and whitespace
                cleaned_fieldnames = {}
                for old_name in reader.fieldnames:
                    new_name = clean_column_name(old_name)
                    cleaned_fieldnames[old_name] = new_name
                print(f"🧹 Cleaning column names: {list(cleaned_fieldnames.values())}")
            
            # Create new row with cleaned column names
            cleaned_row = {}
            for old_name, new_name in cleaned_fieldnames.items():
                # Also clean the cell values to remove any stray BOMs or excess whitespace
                cell_value = row[old_name]
                if isinstance(cell_value, str):
                    cell_value = clean_column_name(cell_value) if cell_value else ""
                cleaned_row[new_name] = cell_value
            rows.append(cleaned_row)
    
    if not rows:
        raise ValueError("CSV file is empty or has no data rows")
    
    print(f"📊 Read {len(rows)} rows from CSV")
    if rows:
        print(f"🔑 Sample row keys: {list(rows[0].keys())}")
        print(f"🏷️ First row model_name: '{rows[0].get('model_name', 'NOT FOUND')}'")
    
    # Initialize JSON structure
    json_data = {}
    
    # Add metadata if preserving
    if preserve_metadata:
        json_data["_metadata"] = {
            "version": "2.0",
            "description": "Converted from CSV format",
            "created": "2025-08-19"
        }
    
    # Always add hardcoded nvidia_nim_prefixes
    json_data["nvidia_nim_prefixes"] = NVIDIA_NIM_PREFIXES
    print(f"🚀 Added hardcoded nvidia_nim_prefixes: {len(json_data['nvidia_nim_prefixes'])} prefixes")
    
    models = {}
    defaults = None
    
    # Process each row
    for i, row in enumerate(rows):
        model_name = row.get("model_name", "").strip()
        
        if model_name == "_defaults":
            # Handle defaults row
            defaults = unflatten_model_config(row)
            print(f"✅ Found _defaults row with {len(defaults)} fields")
        elif model_name:
            # Handle model rows
            models[model_name] = unflatten_model_config(row)
            print(f"✅ Processed model: {model_name}")
        else:
            print(f"⚠️ Skipping row {i+1}: no model_name found")
    
    print(f"📈 Final counts: {len(models)} models, {'1' if defaults else '0'} defaults")
    # Add sections to JSON
    if defaults:
        json_data["_defaults"] = defaults
    
    json_data["models"] = models
    
    # Write JSON file with consistent line endings
    with open(output_file, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    # Normalize line endings to match original file
    try:
        # Detect original file's line endings
        with open(input_file.replace('.csv', '.json'), 'rb') as f:
            sample = f.read(1024)
            if b'\r\n' in sample:
                normalize_line_endings(output_file, '\r\n')
    except FileNotFoundError:
        pass  # Original JSON doesn't exist, keep LF endings
    
    return output_file


def main():
    """Main function for command line usage."""
    if len(sys.argv) not in [2, 3]:
        print("Usage:")
        print("  python convert_model_config.py <input_file.json>     # JSON to CSV")
        print("  python convert_model_config.py <input_file.csv>     # CSV to JSON")
        print("  python convert_model_config.py <input_file> --help  # Show this help")
        print("\nExamples:")
        print("    python convert_model_config.py model_config.json")
        print("    python convert_model_config.py model_config.csv")
        print("\nThe output file will have the same name but opposite extension.")
        sys.exit(1)
    
    if len(sys.argv) == 3 and sys.argv[2] == "--help":
        main()  # Show help and exit
        
    input_file = sys.argv[1]
    
    # Determine conversion direction based on file extension
    input_path = Path(input_file)
    extension = input_path.suffix.lower()
    
    try:
        if extension == '.json':
            # JSON to CSV conversion
            output_file = json_to_csv(input_file)
            print(f"✅ Successfully converted {input_file} to {output_file}")
            
            # Print some statistics
            with open(output_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)
                print(f"📊 Generated CSV with {len(rows)-1} data rows and {len(rows[0])} columns")
                
        elif extension == '.csv':
            # CSV to JSON conversion
            output_file = csv_to_json(input_file)
            print(f"✅ Successfully converted {input_file} to {output_file}")
            
            # Print some statistics
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                models_count = len(data.get("models", {}))
                print(f"📊 Generated JSON with {models_count} models")
                
        else:
            print(f"❌ Unsupported file extension: {extension}")
            print("Supported extensions: .json, .csv")
            sys.exit(1)
            
        print(f"📂 Output saved to: {os.path.abspath(output_file)}")
            
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON format: {e}")
        sys.exit(1)
    except KeyError as e:
        print(f"❌ Missing required key: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()