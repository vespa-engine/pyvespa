import unittest

try:
    import mteb  # noqa: F401
except ImportError:
    mteb = None


@unittest.skipIf(mteb is None, "mteb extra not installed")
class TestMTEBImport(unittest.TestCase):
    def test_module_imports_with_installed_mteb(self):
        # mteb >=2.21 made its dataset type aliases strings; the class body must still evaluate.
        from vespa.evaluation._mteb import VespaMTEBApp, VespaMTEBEvaluator  # noqa: F401
