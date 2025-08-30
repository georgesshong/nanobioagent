#!/usr/bin/env python3
"""Test main.py CLI script and package API integration"""

import unittest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestMainAndPackageIntegration(unittest.TestCase):
    """Test that both main.py script and nanobioagent package work correctly"""
    
    def test_main_script_import(self):
        """Test that main.py CLI script can be imported"""
        try:
            import main
            # Test that it has CLI-related functions
            self.assertTrue(hasattr(main, 'parse_arguments'))
            self.assertTrue(hasattr(main, 'main'))
        except ImportError as e:
            self.fail(f"Could not import main.py: {e}")
    
    def test_package_import(self):
        """Test that nanobioagent package can be imported"""
        try:
            import nanobioagent as nba
            self.assertTrue(hasattr(nba, 'gene_answer'))
        except ImportError as e:
            self.fail(f"Could not import nanobioagent package: {e}")
    
    def test_package_api_functions(self):
        """Test that package API functions exist and are callable"""
        import nanobioagent as nba
        
        # Test main API functions
        self.assertTrue(callable(nba.gene_answer))
        self.assertTrue(callable(nba.answer))
        
        # Test that all methods are available
        self.assertTrue(hasattr(nba, 'gene_answer_direct'))
        self.assertTrue(hasattr(nba, 'gene_answer_agent'))
        self.assertTrue(hasattr(nba, 'gene_answer_genegpt'))
        self.assertTrue(hasattr(nba, 'gene_answer_code'))
        self.assertTrue(hasattr(nba, 'geneGPT_answer_retrieve'))
    
    def test_main_imports_from_package(self):
        """Test that main.py correctly imports from the package"""
        import main
        from nanobioagent.api import gene_answer
        
        # Both should reference the same function
        self.assertEqual(main.gene_answer, gene_answer)
    
    def test_package_mode_detection(self):
        """Test that PACKAGE_MODE is properly set"""
        import main
        
        # Should have PACKAGE_MODE defined
        self.assertTrue(hasattr(main, 'PACKAGE_MODE'))
        self.assertIsInstance(main.PACKAGE_MODE, bool)


if __name__ == "__main__":
    unittest.main()