import unittest
from lxml import etree

from .dynamic import FT, to_xml, attrmap, valmap, create_tag_function, xml_tags, _escape


class TestXMLSchema(unittest.TestCase):
    def test_sanitize_tag_name(self):
        self.assertEqual(FT.sanitize_tag_name("content-node"), "content_node")
        self.assertEqual(FT.sanitize_tag_name("search-engine"), "search_engine")

    def test_restore_tag_name(self):
        self.assertEqual(FT.restore_tag_name("content_node"), "content-node")
        self.assertEqual(FT.restore_tag_name("search_engine"), "search-engine")

    def test_attrmap(self):
        self.assertEqual(attrmap("_max_memory"), "max-memory")
        self.assertEqual(attrmap("content_node"), "content-node")

    def test_valmap(self):
        self.assertEqual(valmap("test"), "test")
        self.assertEqual(valmap([1, 2, 3]), "1 2 3")

    def test_single_tag(self):
        content_tag = FT("content", (), {"attr": "value"})
        xml_output = to_xml(content_tag, indent=False)
        # Expecting the compact format without unnecessary newlines
        self.assertEqual(str(xml_output), '<content attr="value"></content>\n')

    def test_nested_tags(self):
        nested_tag = FT("content", (FT("document", ()),), {"attr": "value"})
        xml_output = to_xml(nested_tag, indent=False)
        # Expecting nested tags with proper newlines and indentation
        expected_output = '<content attr="value"><document></document></content>\n'
        self.assertEqual(str(xml_output), expected_output)

    def test_void_tag(self):
        void_tag = FT("meta", (), void_=True)
        xml_output = to_xml(void_tag, indent=False)
        # Expecting a void tag with a newline at the end
        self.assertEqual(str(xml_output), "<meta />\n")

    def test_tag_with_attributes(self):
        tag_with_attr = FT(
            "content", (), {"max-size": "100", "compression-type": "gzip"}
        )
        xml_output = to_xml(tag_with_attr, indent=False)
        # Expecting the tag with attributes in compact format
        expected_output = '<content max-size="100" compression-type="gzip"></content>\n'
        self.assertEqual(str(xml_output), expected_output)

    def test_escape(self):
        unescaped_text = "<content>"
        self.assertEqual(_escape(unescaped_text), "&lt;content&gt;")

    def test_dynamic_tag_generation(self):
        for tag in xml_tags:
            sanitized_name = FT.sanitize_tag_name(tag)
            tag_func = create_tag_function(tag, False)
            self.assertEqual(tag_func().__class__, FT)
            self.assertEqual(tag_func().tag, sanitized_name)

    def test_tag_addition(self):
        tag = FT("content", (), {"attr": "value"})
        new_tag = FT("document", ())
        tag = tag + new_tag
        self.assertEqual(len(tag.children), 1)
        self.assertEqual(tag.children[0].tag, "document")

    def test_repr(self):
        tag = FT("content", (FT("document", ()),), {"attr": "value"})
        self.assertEqual(repr(tag), "content((document((),{}),),{'attr': 'value'})")


class TestColbertSchema(unittest.TestCase):
    def setUp(self):
        with open("tests/testfiles/relaxng/services.rng", "rb") as schema_file:
            self.relaxng = etree.RelaxNG(etree.parse(schema_file))
        self.xml_schema = """<?xml version="1.0" encoding="utf-8" ?>
    <services version="1.0" xmlns:deploy="vespa" xmlns:preprocess="properties" minimum-required-vespa-version="8.338.38">

        <!-- See https://docs.vespa.ai/en/reference/services-container.html -->
        <container id="default" version="1.0">

            <!-- See https://docs.vespa.ai/en/embedding.html#huggingface-embedder -->
            <component id="e5" type="hugging-face-embedder">
                <transformer-model url="https://huggingface.co/intfloat/e5-small-v2/resolve/main/model.onnx"/>
                <tokenizer-model url="https://huggingface.co/intfloat/e5-small-v2/raw/main/tokenizer.json"/>
                <!-- E5 prompt instructions -->
                <prepend>
                    <query>query:</query>
                    <document>passage:</document>
                </prepend>
            </component>

            <!-- See https://docs.vespa.ai/en/embedding.html#colbert-embedder -->
            <component id="colbert" type="colbert-embedder">
                <transformer-model url="https://huggingface.co/colbert-ir/colbertv2.0/resolve/main/model.onnx"/>
                <tokenizer-model url="https://huggingface.co/colbert-ir/colbertv2.0/raw/main/tokenizer.json"/>
            </component>

            <document-api/>
            <search/>
            <nodes count="1">
                <resources vcpu="4" memory="16Gb" disk="125Gb">
                    <gpu count="1" memory="16Gb"/>
                </resources>
            </nodes>
            
        </container>

        <!-- See https://docs.vespa.ai/en/reference/services-content.html -->
        <content id="text" version="1.0">
            <min-redundancy>2</min-redundancy>
            <documents>
                <document type="doc" mode="index" />
            </documents>
            <nodes count="2"/>
        </content>

    </services>
    """

    def test_valid_colbert_schema(self):
        to_validate = etree.parse("tests/testfiles/services/colbert/services.xml")
        # Validate against relaxng
        self.relaxng.validate(to_validate)
        self.assertTrue(self.relaxng.validate(to_validate))

    def test_valid_schema_from_string(self):
        to_validate = etree.fromstring(self.xml_schema.encode("utf-8"))
        self.assertTrue(self.relaxng.validate(to_validate))

    def test_invalid_schema_from_string(self):
        invalid_xml = self.xml_schema.replace("document-api", "document-api-")
        to_validate = etree.fromstring(invalid_xml.encode("utf-8"))
        self.assertFalse(self.relaxng.validate(to_validate))


if __name__ == "__main__":
    unittest.main()
