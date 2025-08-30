# NanoBioAgent (NBA): Nano-Scale Language Model Agents for Genomics

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Enabling Small Language Models to Excel at Biological Question Answering Through Intelligent Tool Orchestration**

NanoBioAgent (NBA) is a agent-based framework that democratizes genomics AI by enabling small language models to achieve performance comparable to much larger models through intelligent tool orchestration. Built upon insights from GeneGPT, NBA introduces an agentic architecture that reduces model requirements significantly while maintaining accuracy through smart task decomposition and NCBI API integration.

## 🎯 Key Features

- **🔬 Nano-Scale Intelligence**: Achieve large model performance (e.g. closed models like ChatGPT and Claude) with small models (e.g. nemotron-nano-8b or qwen 2.5 7b)
- **🤖 Agentic Architecture**: Modular, configuration-driven framework using LangChain LCEL
- **🔄 Multiple Methods for Benchmarks**: NBA vs GeneGPT vs Direct LLM vs Function Call comparison
- **🌐 Model Agnostic**: Support for 30+ models across OpenAI, Anthropic, Google, Ollama, NVIDIA NIM, and HuggingFace  
- **⚡ Performance Optimized**: Intelligent API caching, smart truncation, and comprehensive monitoring
- **📊 Comprehensive Evaluation**: Built-in benchmarking against GeneTuring dataset with detailed analytics

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/georgesshong/GeneGPT-main.git
cd GeneGPT-main

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Basic Usage

```bash
# Run with NBA (NanoBioAgent framework)
python main.py 111111 --method agent --model gpt-4o-mini

# Compare with original GeneGPT implementation  
python main.py 111111 --method genegpt --model gpt-3.5-turbo

# Use nano-scale (7 or 8 bil param) model
python main.py 111111 --method agent --model qwen_qwen2.5-coder-7b-instruct
python main.py 111111 --method agent --model llama-3.1-nemotron-nano-8b-v1

# Use small local models via Ollama (needs local installation)
python main.py 111111 --method genegpt --model ollama/meditron:7b

# Run specific tasks only (0-indexed)
python main.py 111111 --method agent --task_idx "0,2,4"
```

### Quick Test

```python
from main import gene_answer

# Ask a genomics question with NBA
result = gene_answer(
    question="What is the official gene symbol of LMP10?",
    method="agent",  # NBA framework
    model_name="gpt-4o-mini"  # Small model!
)

print(f"Answer: {result[2]}")  # Output: "PSMB10"
```

## 📈 Performance Results

| Method | Overall Score | Model Requirements | Num Parameters |
|--------|---------------|-------------------|------------------|
| **NBA (NanoBioAgent)** | **0.76-0.87** | **Nano/Small Models** | **3 to 12 bil** |
| GeneGPT (Original) | 0.83 | Large (GPT-3.5 / code-davinci-002) | ~175 bil and above |
| Direct LLM | 0.45 | Large (GPT-4 and above) | ~175 bil and above |
| Function-Based | 0.9-1.0 | N.A. | N.A. |

*Results on GeneTuring benchmark (9 genomics tasks, 450 questions)*

### 🔬 **NBA's Nano-Scale Achievement:**
- **gpt-4.1-nano** (NBA) >= **gpt-4.1** (GeneGPT) performance
- **claude-3-5-haiku** (NBA) > **claude-3-7-sonnet** (GeneGPT) performance  
- **qwen_qwen2.5-coder-7b-instruct** achieves large model performance with sub-10b parameters

## 🏗️ Architecture Overview

### GeneGPT (Original Implementation)
```
Question → Large Prompt → GPT-3.5-turbo-16k → API Parsing → Answer
```

### NBA (NanoBioAgent Framework) 
```
Question → Task Classification → Parameter Inference → Tool Execution → Answer Parsing → Answer
            ↓ Nano Models         ↓ Small Models        ↓ Cached APIs    ↓ Smart Parsing
            Pattern + LLM         Few-shot Examples     Retry Logic      Field Truncation
```

**NBA's Innovation**: Breaks complex genomics queries into nano-scale operations, allowing small models to achieve large model performance through intelligent orchestration and domain-specific optimization.

## 📊 Supported Tasks

| Task Category | Examples | Supported APIs |
|---------------|----------|----------------|
| **Gene Nomenclature** | Gene alias, name conversion | NCBI Gene |
| **Genomic Location** | Gene/SNP chromosomal location | NCBI Gene, dbSNP |
| **Functional Analysis** | Gene-disease associations, protein coding | NCBI Gene, OMIM |
| **Sequence Alignment** | DNA sequence mapping | NCBI BLAST |

## 🛠️ Methods Comparison

### 🏆 **NBA Method** (Recommended)
- **Best for**: Production use, nano/small models, cost efficiency, interpretability
- **Models**: nemotron-nano-8b, gpt-4.1-nano, claude-3-5-haiku
- **Pros**: Nano-scale intelligence, modular, efficient, extensible, cost-effective
- **Cons**: Slightly more complex setup

### 🔄 **GeneGPT Method** (Baseline)
- **Best for**: Easy implementation, research replication
- **Models**: gpt-4, claude-3-7-sonnet
- **Pros**: Simple setup, established baseline
- **Cons**: Brittle, requires large models, higher token usage, costly

### 💡 **Direct Method**
- **Best for**: Quick testing, model knowledge assessment, general set-up
- **Models**: Any (large) model, tests pure model knowledge
- **Pros**: No API dependencies, direct execution, pure LLM assessment
- **Cons**: Limited accuracy for specialized queries, no tool augmentation

### 🔍 **Function Retrieve Method**
- **Best for**: Specific use, privacy-sensitive applications, embedding comparison
- **Models**: sentence transformers to match limited function calls
- **Pros**: Coded API function calls good at what it does when applicable
- **Cons**: limited reasoning and small scope

## 🔧 Configuration

NBA is fully configuration-driven, enabling rapid iteration without code changes:

```
config/
├── tasks.json              # Task definitions and patterns
├── plans.json              # Execution plan definitions  
├── tools/
│   └── ncbi_tools.json     # NCBI tool configurations
└── examples/
    └── ncbi_examples.json  # Few-shot learning examples
```

Example task configuration:
```json
{
  "gene_alias": {
    "description": "Find official gene symbols",
    "patterns": ["symbol of (.+)", "symbol for (.+)"],
    "plan": "ncbi_simple_search",
    "answer_format": {"type": "string", "extract": "gene_symbol"}
  }
}
```

**NBA's Advantage**: Configuration-driven design allows domain experts to enhance performance without touching code, while the nano-scale operations ensure even small models can execute complex multi-step reasoning.

## 📝 Evaluation & Benchmarking

### Run Evaluation
```bash
# Evaluate single approach
python evaluate.py results/111111_gpt-4o-mini_agent/

# Compare multiple approaches
python compare.py \
    results/111111_gpt-4o-mini_agent/ \
    results/111111_gpt-4o-mini_genegpt/ \
    results/111111_claude-3-5-sonnet_agent/
```

### Evaluation Metrics
- **Task-specific accuracy** for each genomics domain
- **API call efficiency** and caching effectiveness
- **Token usage** and cost analysis
- **Error categorization** and failure mode analysis

## 🌐 Supported Models

### Cloud APIs
- **OpenAI**: GPT-4o, GPT-4o-mini, GPT-3.5-turbo
- **Anthropic**: Claude-3.5-Sonnet, Claude-4-Sonnet
- **Google**: Gemini-2.0-flash, Gemini-1.5-flash
- **NVIDIA NIM**: Llama-3.1-Nemotron, DeepSeek-R1

### Local Models
- **Ollama**: Llama3, Meditron, Gemma3, CodeLlama
- **HuggingFace**: Any compatible model with transformers

### Model Selection Guide
```python
# For nano-scale efficiency (NBA's specialty)
model_name = "gpt-4o-mini"

# For local nano-scale deployment
model_name = "ollama/meditron:7b"

# For maximum accuracy (GeneGPT baseline)
model_name = "gpt-4o"

# For cost-conscious production
model_name = "claude-3-haiku"

# For privacy/local use
model_name = "ollama/llama3:8b"
```

## 🧪 Research Applications

### Academic Research
- **Efficiency Studies**: Compare nano vs large model performance across domains
- **Agentic Architecture**: Investigate tool orchestration vs monolithic approaches
- **Cost-Performance Analysis**: Quantify accuracy/cost trade-offs with NBA
- **Ablation Studies**: Modular design enables component-wise testing

### Industry Applications  
- **Pharmaceutical R&D**: Gene-drug interaction analysis with cost efficiency
- **Clinical Genomics**: Patient variant interpretation using local nano models
- **Academic Labs**: Research-grade genomics AI within limited budgets

### Educational Use
- **Bioinformatics Training**: Hands-on NCBI API learning with affordable models
  - **AI in Science**: Showcase of domain-specific small model optimization
- **Software Engineering**: Modern Python/LangChain architecture patterns
- **Cost-Conscious AI**: Demonstrate enterprise AI without large model expenses

## 📚 Documentation

| Document | Description |
|----------|-------------|
| **[🏗️ NBA Architecture](docs/NBA_ARCHITECTURE.md)** | Complete architecture overview and component relationships |
| **[🔍 Technical Review](docs/NBA_TECHNICAL_REVIEW.md)** | Comprehensive technical analysis and performance assessment |
| **[⚖️ NBA vs GeneGPT](docs/NBA_VS_GENEGPT.md)** | Detailed comparison of approaches and innovations |
| **[🚀 API Reference](docs/API_REFERENCE.md)** | Detailed function and class documentation |
| **[🛠️ Contributing Guide](docs/CONTRIBUTING.md)** | Development guidelines and extension examples |

## 🤝 Contributing

We welcome contributions! See our [Contributing Guide](docs/CONTRIBUTING.md) for details.

### Quick Contribution Areas
- **New Tasks**: Add support for additional genomics queries
- **Nano Model Optimization**: Improve small model performance
- **Local Model Integration**: Support for new local/edge models
- **Performance Optimization**: Caching and truncation improvements
- **Cost Analysis**: Model efficiency and cost-performance metrics

### Development Setup
```bash
# Development installation
pip install -e .
pip install -r requirements.txt

# Run tests
python -m tests/

# Run code quality checks
black .
flake8 .
mypy .
```

## 📄 Citation (TO BE UPDATED!)

If you use NanoBioAgent (NBA) in your research, please cite:
(Placeholder below only)
```bibtex

@misc{nbaHong2025,
  title={Nano Bio-Agents: Small Language Model Agents in Genomics Applications},
  author={George Shaw-Shiun Hong},
  year={2025},
  eprint={XXXX.XXXXX},
  archivePrefix={arXiv},
  primaryClass={cs.AI}
}
```

## 🔗 Related Work

- **[GeneGPT Original](https://github.com/ncbi/GeneGPT)**: Foundation work for LLM-genomics integration
- **[GeneTuring Benchmark](https://github.com/ncbi/geneturing)**: Genomics question answering dataset
- **[NCBI E-utilities](https://www.ncbi.nlm.nih.gov/books/NBK25497/)**: NCBI API documentation
- **[LangChain](https://github.com/langchain-ai/langchain)**: Framework for LLM applications

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/georgesshong/nanobioagent/issues)
- **Discussions**: [GitHub Discussions](https://github.com/georgesshong/nanobioagent/discussions)
- **Email**: [gehong@ethz.ch](mailto:gehong@ethz.ch)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Daniel Trejo Banos**: For the ideas and advice supporting the research
- **GeneGPT Team**: For the foundational work that inspired NBA and their open-sourced code
- **NCBI**: For providing comprehensive genomics APIs
- **LangChain Community**: For the excellent framework enabling agentic AI
- **Open Source Contributors**: For model implementations and tools

---

<div align="center">

**[⭐ Star this repo](https://github.com/georgesshong/nanobioagent)** if you find NBA useful!

</div>