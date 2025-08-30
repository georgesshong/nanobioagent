#!/usr/bin/env python3
"""Test basic functionality without API calls"""

import unittest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestBasicFunctionality(unittest.TestCase):
    """Test basic functions work without external dependencies"""
    
    def test_log_event_function(self):
        """Test log_event function works"""
        from nanobioagent.tools.gene_utils import log_event
        
        # Should not raise an exception
        try:
            log_event("Test message")
        except Exception as e:
            self.fail(f"log_event failed: {e}")
    
    def test_prompt_functions(self):
        """Test prompt generation functions"""
        from nanobioagent.core.prompts import get_prompt_header
        
        try:
            # Test with a simple mask
            mask = [True, False, True, False, True, False] 
            prompt = get_prompt_header(mask)
            self.assertIsInstance(prompt, str)
            self.assertGreater(len(prompt), 0)
        except Exception as e:
            self.fail(f"Prompt function failed: {e}")
    
    def test_framework_initialization(self):
        """Test AgentFramework can be initialized"""
        try:
            from nanobioagent.core.agent_framework import AgentFramework
            
            # This might fail due to missing config, but should not crash on import
            framework = AgentFramework()
            self.assertIsNotNone(framework)
        except FileNotFoundError:
            # Expected if config files are missing
            self.skipTest("Config files not found - expected in development")
        except ImportError as e:
            self.fail(f"Framework initialization import failed: {e}")


if __name__ == "__main__":
    unittest.main()