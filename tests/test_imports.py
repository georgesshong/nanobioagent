#!/usr/bin/env python3
"""Test that all package imports work correctly"""

import unittest
import sys
from pathlib import Path

# Add parent directory to path so we can import nanobioagent
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestImports(unittest.TestCase):
    """Test all critical imports work"""
    
    def test_tools_imports(self):
        """Test tools module imports"""
        try:
            from nanobioagent.tools.gene_utils import call_api, log_event
            import nanobioagent.tools.ncbi_tools as ncbi_tools
            from nanobioagent.tools import ncbi_query
        except ImportError as e:
            self.fail(f"Tools import failed: {e}")
    
    def test_core_imports(self):
        """Test core module imports"""
        try:
            from nanobioagent.core.model_utils import create_langchain_model
            from nanobioagent.core.prompts import get_prompt_header
            from nanobioagent.core.agent_framework import AgentFramework
        except ImportError as e:
            self.fail(f"Core import failed: {e}")
    
    def test_config_access(self):
        """Test that config files can be accessed"""
        try:
            from nanobioagent.core.agent_framework import AgentFramework
            # This will try to load config files
            framework = AgentFramework()
            self.assertIsNotNone(framework)
        except Exception as e:
            self.fail(f"Config access failed: {e}")


if __name__ == "__main__":
    unittest.main()