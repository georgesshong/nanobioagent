# Contributing to NanoBioAgent (NBA) 🤝

Thank you very much for your interest in contributing to NanoBioAgent! This guide will help you get started.

## 🌟 Ways to Contribute

- **🔧 Code Contributions**: Fix bugs or add new features
- **💡 Feature Requests**: Have ideas for improvements?
- **📝 Documentation**: Help improve our docs and examples
- **🧪 Testing**: Add tests or test on different platforms
- **📊 Benchmarking**: Contribute performance analysis or new datasets

## 🚀 Quick Start for Contributors

### Prerequisites
- Python 3.9+ (Tested with 3.9.13 on Windows)
- Git
- API keys for testing (e.g. HuggingFace, Nvidia, OpenAI, Anthropic, Google etc. You just need the ones for the LLM providers you want to run on.)

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/georgesshong/nanobioagent.git
   cd nanobioagent
   ```

2. **Create Development Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e .
   pip install -r requirements.txt  # When we create this file
   ```

3. **Set Up Environment Variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Verify Setup**
   ```bash
   python -c "import nanobioagent; print('Setup successful!')"
   python main.py --help
   ```

## 🐛 Reporting Bugs

**Before submitting**, please check:
- [ ] Search existing [issues](https://github.com/georgesshong/nanobioagent/issues)
- [ ] Try with the latest version
- [ ] Include error messages and logs

### Bug Report Template
```markdown
**Bug Description**: Clear description of the issue

**Steps to Reproduce**:
1. Run this command: `python main.py ...`
2. With this input: `...`
3. See error: `...`

**Expected Behavior**: What should have happened?

**Environment**:
- OS: [Windows]
- Python: [3.9+]


## 💡 Feature Requests

We welcome ideas for:
- **New genomics tasks** and query types
- **Additional model providers** (local models, new APIs)
- **Performance optimizations** for small models
- **Better evaluation metrics** and benchmarks
- **Integration improvements** with existing workflows

### Feature Request Template
```markdown
**Feature Description**: Brief description of the requested feature

**Use Case**: Why is this feature needed? What problem does it solve?

**Proposed Solution**: How do you envision this working?

**Additional Context**: Any relevant examples, papers, or references
```

## 🔧 Code Contributions

### Development Workflow

1. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-number
   ```

2. **Make Changes**
   - Follow existing code style
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Thoroughly**
   ```bash
   # Run basic tests
   python -m pytest tests/
   
   # Test with different models
   python main.py test --method agent --model gpt-4o-mini
   python main.py test --method agent --model qwen_qwen2.5-coder-7b-instruct
   ```

4. **Commit and Push**
   ```bash
   git add .
   git commit -m "feat: add support for new genomics task"
   git push origin feature/your-feature-name
   ```

5. **Create Pull Request**
   - Use descriptive title and description
   - Reference any related issues
   - Include testing information


## 🧪 Testing Guidelines

### Test Types We Need
- **Unit tests**: Test individual functions
- **Integration tests**: Test API calls and workflows  
- **Model tests**: Test with different LLMs
- **Performance tests**: Benchmark latency and accuracy

### Running Tests
```bash
# Specific test file
python -m tests/test_main_integration.py

```

## 📝 Documentation Contributions

Documentation improvements are highly valued:

- **README improvements**: Clearer explanations, better examples
- **Code documentation**: Docstrings, inline comments
- **Configuration guides**: Help with complex setups
- **Tutorials**: Step-by-step guides for specific use cases
- **API documentation**: Reference materials

## 🎯 Priority Contribution Areas

### 🔥 High Priority
- **AlphaGenome integration**: continue to explore AlphaGenome as tools
- **Nano model optimization**: Improve performance for even small models
- **Local model support**: Better Ollama/HuggingFace integration for local clusters
- **Error handling**: More robust error messages and recovery

### 📊 Medium Priority
- **New genomics tasks**: Expand beyond current 9 task types
- **Visualization tools**: Result plotting and analysis
- **Configuration validation**: Better config error messages
- **Documentation examples**: More real-world use cases

### 💭 Ideas Welcome
- **What other Bio-Informatics tasks can we try?**: AlphaGenome or other potential tasks
- **What other Cost optimization can we explore?**: Reduce API call costs
- **What GUI interface may help testing and research?**: Web or desktop interface

## 📞 Getting Help

- **GitHub Discussions**: Best for questions and brainstorming
- **GitHub Issues**: For bug reports and feature requests
- **Email**: [gehong@ethz.ch](mailto:gehong@ethz.ch) for complex questions

## 🏆 Recognition

Contributors will be:
- Listed in README.md acknowledgments
- Mentioned in release notes
- Credited in any related publications (for significant contributions)

## 📋 Pull Request Checklist

Before submitting, ensure:
- [ ] Code follows project style guidelines
- [ ] Tests pass locally
- [ ] Documentation updated (if needed)
- [ ] Commit messages are descriptive
- [ ] PR description explains the changes
- [ ] Related issues are referenced

## 🎉 Thank You!

Every contribution, no matter how small, helps make NanoBioAgent better for the genomics research community. We appreciate your time and effort!

---

*This project builds upon the excellent work of NCBI's GeneGPT team and the broader open-source community.*