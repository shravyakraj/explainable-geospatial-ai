import unittest
import importlib

class TestPackageFoundation(unittest.TestCase):
    def test_package_metadata(self):
        """Verify geospatial_ai package structure and version definition."""
        import geospatial_ai
        self.assertTrue(hasattr(geospatial_ai, "__version__"))
        self.assertEqual(geospatial_ai.__version__, "0.1.0")

if __name__ == "__main__":
    unittest.main()
