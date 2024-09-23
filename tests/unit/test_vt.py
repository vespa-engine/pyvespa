import unittest
from vespa.configuration.vt import compare_xml


class TestXMLComparison(unittest.TestCase):
    def test_equal_simple(self):
        xml1 = "<root><child>Text</child></root>"
        xml2 = "<root><child>Text</child></root>"
        self.assertTrue(compare_xml(xml1, xml2))

    def test_whitespace_differences(self):
        xml1 = "<root><child>Text</child></root>"
        xml2 = "<root>\n  <child>Text</child>\n</root>"
        self.assertTrue(compare_xml(xml1, xml2))

    def test_attribute_order(self):
        xml1 = '<root><child b="2" a="1">Text</child></root>'
        xml2 = '<root><child a="1" b="2">Text</child></root>'
        self.assertTrue(compare_xml(xml1, xml2))

    def test_text_whitespace(self):
        xml1 = "<root><child> Text </child></root>"
        xml2 = "<root><child>Text</child></root>"
        self.assertTrue(compare_xml(xml1, xml2))

    def test_different_text(self):
        xml1 = "<root><child>Text1</child></root>"
        xml2 = "<root><child>Text2</child></root>"
        self.assertFalse(compare_xml(xml1, xml2))

    def test_different_structure(self):
        xml1 = "<root><child>Text</child></root>"
        xml2 = "<root><child><subchild>Text</subchild></child></root>"
        self.assertFalse(compare_xml(xml1, xml2))

    def test_namespace_handling(self):
        xml1 = '<root xmlns="namespace"><child>Text</child></root>'
        xml2 = "<root><child>Text</child></root>"
        # Namespaces are considered in the tag comparison
        self.assertFalse(compare_xml(xml1, xml2))

    def test_comments_ignored(self):
        xml1 = "<root><!-- A comment --><child>Text</child></root>"
        xml2 = "<root><child>Text</child></root>"
        # Comments are not part of the element tree; they are ignored
        self.assertTrue(compare_xml(xml1, xml2))

    def test_processing_instructions(self):
        xml1 = "<?xml version='1.0'?><root><child>Text</child></root>"
        xml2 = "<root><child>Text</child></root>"
        self.assertTrue(compare_xml(xml1, xml2))

    def test_different_attributes(self):
        xml1 = '<root><child a="1">Text</child></root>'
        xml2 = '<root><child a="2">Text</child></root>'
        self.assertFalse(compare_xml(xml1, xml2))

    def test_additional_attributes(self):
        xml1 = '<root><child a="1" b="2">Text</child></root>'
        xml2 = '<root><child a="1">Text</child></root>'
        self.assertFalse(compare_xml(xml1, xml2))

    def test_multiple_children_order(self):
        xml1 = "<root><child>1</child><child>2</child></root>"
        xml2 = "<root><child>2</child><child>1</child></root>"
        self.assertTrue(compare_xml(xml1, xml2))

    def test_namespace_prefixes(self):
        xml1 = '<root xmlns:ns="namespace"><ns:child>Text</ns:child></root>'
        xml2 = "<root><child>Text</child></root>"
        # Different namespaces make the tags different
        self.assertFalse(compare_xml(xml1, xml2))

    def test_cdata_handling(self):
        xml1 = "<root><child><![CDATA[Text]]></child></root>"
        xml2 = "<root><child>Text</child></root>"
        self.assertTrue(compare_xml(xml1, xml2))


if __name__ == "__main__":
    unittest.main()
