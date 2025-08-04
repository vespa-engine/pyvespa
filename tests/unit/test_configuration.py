import unittest
from lxml import etree
import xml.etree.ElementTree as ET
from vespa.configuration.vt import (
    VT,
    vt,
    create_tag_function,
    attrmap,
    valmap,
    to_xml,
    compare_xml,
    vt_escape,
)

from vespa.configuration.services import *
from vespa.configuration.query_profiles import *


class TestVT(unittest.TestCase):
    def test_sanitize_tag_name(self):
        self.assertEqual(VT.sanitize_tag_name("content-node"), "content_node")
        self.assertEqual(VT.sanitize_tag_name("from"), "from_")

    def test_restore_tag_name(self):
        self.assertEqual(vt("content_node").restore_tag_name(), "content-node")
        self.assertEqual(vt("search_engine").restore_tag_name(), "search-engine")

    def test_restore_with_underscores(self):
        self.assertEqual(
            vt("content_node", replace_underscores=False).tag, "content_node"
        )
        self.assertEqual(
            vt("search_engine", replace_underscores=False).tag, "search_engine"
        )

    def test_attrmap(self):
        self.assertEqual(attrmap("_max_memory"), "max-memory")
        self.assertEqual(attrmap("content_node"), "content-node")

    def test_valmap(self):
        self.assertEqual(valmap("test"), "test")
        self.assertEqual(valmap([1, 2, 3]), "1 2 3")

    def test_single_tag(self):
        content_tag = VT("content", (), {"attr": "value"})
        xml_output = to_xml(content_tag, indent=False)
        # Expecting the compact format without unnecessary newlines
        self.assertEqual(str(xml_output), '<content attr="value"></content>')

    def test_nested_tags(self):
        nested_tag = vt("content", attr="value")(vt("document"))
        xml_output = to_xml(nested_tag, indent=False)
        # Expecting nested tags with proper newlines and indentation
        expected_output = '<content attr="value"><document></document></content>'
        self.assertEqual(str(xml_output), expected_output)

    def test_nested_tags_with_text(self):
        nested_tag = VT("content", (VT("document", ("text",)),), {"attr": "value"})
        xml_output = to_xml(nested_tag, indent=False)
        # Expecting nested tags with proper newlines and indentation
        parsed_output = ET.fromstring(str(xml_output))
        self.assertEqual(parsed_output.tag, "content")
        self.assertEqual(parsed_output.attrib, {"attr": "value"})
        self.assertEqual(parsed_output[0].tag, "document")
        self.assertEqual(parsed_output[0].text, "text")

    def test_void_tag(self):
        void_tag = VT("meta", (), void_=True)
        xml_output = to_xml(void_tag, indent=False)
        # Expecting a void tag with a newline at the end
        self.assertEqual(str(xml_output), "<meta />")

    def test_tag_with_attributes(self):
        tag_with_attr = VT(
            "content", (), {"max-size": "100", "compression-type": "gzip"}
        )
        xml_output = to_xml(tag_with_attr, indent=False)
        # Expecting the tag with attributes in compact format
        expected_output = '<content max-size="100" compression-type="gzip"></content>'
        self.assertEqual(str(xml_output), expected_output)

    def test_escape(self):
        unescaped_text = "<content>"
        self.assertEqual(vt_escape(unescaped_text), "&lt;content&gt;")

    def test_dynamic_tag_generation(self):
        for tag in services_tags:
            sanitized_name = VT.sanitize_tag_name(tag)
            tag_func = create_tag_function(tag, False)
            self.assertEqual(tag_func().__class__, VT)
            self.assertEqual(tag_func().tag, sanitized_name)

    def test_tag_addition(self):
        tag = VT("content", (), {"attr": "value"})
        new_tag = VT("document", ())
        tag = tag + new_tag
        self.assertEqual(len(tag.children), 1)
        self.assertEqual(tag.children[0].tag, "document")

    def test_repr(self):
        tag = VT("content", (VT("document", ()),), {"attr": "value"})
        self.assertEqual(repr(tag), "content((document((),{}),),{'attr': 'value'})")


class TestColbertServiceConfiguration(unittest.TestCase):
    def setUp(self):
        self.xml_file_path = "tests/testfiles/services/colbert/services.xml"
        self.xml_schema = """<?xml version="1.0" encoding="utf-8" ?>
    <services version="1.0" minimum-required-vespa-version="8.338.38">

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
        to_validate = etree.parse(self.xml_file_path)
        # Validate against relaxng
        self.assertTrue(validate_services(to_validate))

    def test_valid_schema_from_string(self):
        to_validate = etree.fromstring(self.xml_schema.encode("utf-8"))
        self.assertTrue(validate_services(to_validate))

    def test_invalid_schema_from_string(self):
        invalid_xml = self.xml_schema.replace("document-api", "asdf")
        to_validate = etree.fromstring(invalid_xml.encode("utf-8"))
        self.assertFalse(validate_services(to_validate))

    def test_generate_colbert_services(self):
        self.maxDiff = None
        # Generated XML using dynamic tag functions
        generated_services = services(
            container(id="default", version="1.0")(
                component(id="e5", type="hugging-face-embedder")(
                    transformer_model(
                        url="https://huggingface.co/intfloat/e5-small-v2/resolve/main/model.onnx"
                    ),
                    tokenizer_model(
                        url="https://huggingface.co/intfloat/e5-small-v2/raw/main/tokenizer.json"
                    ),
                    prepend(query("query:"), document("passage:")),
                ),
                component(id="colbert", type="colbert-embedder")(
                    transformer_model(
                        url="https://huggingface.co/colbert-ir/colbertv2.0/resolve/main/model.onnx"
                    ),
                    tokenizer_model(
                        url="https://huggingface.co/colbert-ir/colbertv2.0/raw/main/tokenizer.json"
                    ),
                ),
                document_api(),
                search(),
                nodes(count="1")(
                    resources(vcpu="4", memory="16Gb", disk="125Gb")(
                        gpu(count="1", memory="16Gb")
                    )
                ),
            ),
            content(id="text", version="1.0")(
                min_redundancy("2"),
                documents(document(type="doc", mode="index")),
                nodes(count="2"),
            ),
            version="1.0",
            minimum_required_vespa_version="8.338.38",
        )
        generated_xml = generated_services.to_xml()
        # Validate against relaxng
        self.assertTrue(validate_services(etree.fromstring(str(generated_xml))))
        self.assertTrue(compare_xml(self.xml_schema, str(generated_xml)))


class TestBillionscaleServiceConfiguration(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.xml_file_path = "tests/testfiles/services/billion-scale-image-search/src/main/application/services.xml"
        self.xml_schema = """<?xml version="1.0" encoding="utf-8" ?>
<services version='1.0' xmlns:deploy="vespa" xmlns:preprocess="properties">

  <container id='default' version='1.0'>
    <nodes count='1'/>
    <component id='ai.vespa.examples.Centroids' bundle='billion-scale-image-search'/>
    <component id='ai.vespa.examples.DimensionReducer' bundle='billion-scale-image-search'/>
    <component id="ai.vespa.examples.BPETokenizer" bundle='billion-scale-image-search'>
      <config name="ai.vespa.examples.bpe-tokenizer">
        <contextlength>77</contextlength>
        <vocabulary>files/bpe_simple_vocab_16e6.txt.gz</vocabulary>
      </config>
    </component>
    <model-evaluation>
      <onnx>
        <models>
          <model name="text_transformer">
            <intraop-threads>1</intraop-threads>
          </model>
          <model name="vespa_innerproduct_ranker">
            <intraop-threads>1</intraop-threads>
          </model>
        </models>
      </onnx>
    </model-evaluation>
    <search>
      <chain id='default' inherits='vespa'>
        <searcher id='ai.vespa.examples.searcher.DeDupingSearcher' bundle='billion-scale-image-search'/>
        <searcher id='ai.vespa.examples.searcher.RankingSearcher' bundle='billion-scale-image-search'/>
        <searcher id="ai.vespa.examples.searcher.CLIPEmbeddingSearcher" bundle="billion-scale-image-search"/>
        <searcher id='ai.vespa.examples.searcher.SPANNSearcher' bundle='billion-scale-image-search'/>
      </chain>
    </search>
    <document-api/>
    <document-processing>
      <chain id='neighbor-assigner' inherits='indexing'>
        <documentprocessor id='ai.vespa.examples.docproc.DimensionReductionDocProc'
                           bundle='billion-scale-image-search'/>
        <documentprocessor id='ai.vespa.examples.docproc.AssignCentroidsDocProc'
                           bundle='billion-scale-image-search'/>
      </chain>
    </document-processing>
  </container>

  <content id='graph' version='1.0'>
    <min-redundancy>1</min-redundancy>
    <documents>
      <document mode='index' type='centroid'/>
      <document-processing cluster='default' chain='neighbor-assigner'/>
    </documents>
    <nodes count='1'/>
    <engine>
      <proton>
        <tuning>
          <searchnode>
            <feeding>
              <concurrency>1.0</concurrency>
            </feeding>
          </searchnode>
        </tuning>
      </proton>
    </engine>
  </content>

  <content id='if' version='1.0'>
    <min-redundancy>1</min-redundancy>
    <documents>
      <document mode='index' type='image'/>
      <document-processing cluster='default' chain='neighbor-assigner'/>
    </documents>
    <nodes count='1'/>
    <engine>
      <proton>
        <tuning>
          <searchnode>
            <requestthreads>
              <persearch>2</persearch>
            </requestthreads>
            <feeding>
              <concurrency>1.0</concurrency>
            </feeding>
            <summary>
              <io>
                <read>directio</read>
              </io>
              <store>
                <cache>
                  <maxsize-percent>5</maxsize-percent>
                  <compression>
                    <type>lz4</type>
                  </compression>
                </cache>
                <logstore>
                  <chunk>
                    <maxsize>16384</maxsize>
                    <compression>
                      <type>zstd</type>
                      <level>3</level>
                    </compression>
                  </chunk>
                </logstore>
              </store>
            </summary>
          </searchnode>
        </tuning>
      </proton>
    </engine>
  </content>
</services>
"""

    def test_valid_billion_scale_config(self):
        to_validate = etree.parse(self.xml_file_path)
        # Validate against relaxng
        self.assertTrue(validate_services(to_validate))

    def test_config_from_string(self):
        to_validate = etree.fromstring(self.xml_schema.encode("utf-8"))
        self.assertTrue(validate_services(to_validate))

    def test_generate_billion_scale_services(self):
        # Generated XML using dynamic tag functions
        generated_services = services(
            container(id="default", version="1.0")(
                nodes(count="1"),
                component(
                    id="ai.vespa.examples.Centroids",
                    bundle="billion-scale-image-search",
                ),
                component(
                    id="ai.vespa.examples.DimensionReducer",
                    bundle="billion-scale-image-search",
                ),
                component(
                    id="ai.vespa.examples.BPETokenizer",
                    bundle="billion-scale-image-search",
                )(
                    config(name="ai.vespa.examples.bpe-tokenizer")(
                        vt(
                            "contextlength", "77"
                        ),  # using vt as this is not a predefined tag
                        vt(
                            "vocabulary", "files/bpe_simple_vocab_16e6.txt.gz"
                        ),  # using vt as this is not a predefined tag
                    ),
                ),
                model_evaluation(
                    onnx(
                        models(
                            model(name="text_transformer")(intraop_threads("1")),
                            model(name="vespa_innerproduct_ranker")(
                                intraop_threads("1")
                            ),
                        ),
                    ),
                ),
                search(
                    chain(id="default", inherits="vespa")(
                        searcher(
                            id="ai.vespa.examples.searcher.DeDupingSearcher",
                            bundle="billion-scale-image-search",
                        ),
                        searcher(
                            id="ai.vespa.examples.searcher.RankingSearcher",
                            bundle="billion-scale-image-search",
                        ),
                        searcher(
                            id="ai.vespa.examples.searcher.CLIPEmbeddingSearcher",
                            bundle="billion-scale-image-search",
                        ),
                        searcher(
                            id="ai.vespa.examples.searcher.SPANNSearcher",
                            bundle="billion-scale-image-search",
                        ),
                    ),
                ),
                document_api(),
                document_processing(
                    chain(id="neighbor-assigner", inherits="indexing")(
                        documentprocessor(
                            id="ai.vespa.examples.docproc.DimensionReductionDocProc",
                            bundle="billion-scale-image-search",
                        ),
                        documentprocessor(
                            id="ai.vespa.examples.docproc.AssignCentroidsDocProc",
                            bundle="billion-scale-image-search",
                        ),
                    ),
                ),
            ),
            content(id="graph", version="1.0")(
                min_redundancy("1"),
                documents(
                    document(mode="index", type="centroid"),
                    document_processing(cluster="default", chain="neighbor-assigner"),
                ),
                nodes(count="1"),
                engine(
                    proton(
                        tuning(
                            searchnode(
                                feeding(concurrency("1.0")),
                            ),
                        ),
                    ),
                ),
            ),
            content(id="if", version="1.0")(
                min_redundancy("1"),
                documents(
                    document(mode="index", type="image"),
                    document_processing(cluster="default", chain="neighbor-assigner"),
                ),
                nodes(count="1"),
                engine(
                    proton(
                        tuning(
                            searchnode(
                                requestthreads(persearch("2")),
                                feeding(concurrency("1.0")),
                                summary(
                                    io_(read("directio")),
                                    store(
                                        cache(
                                            maxsize_percent("5"),
                                            compression(
                                                type_("lz4")
                                            ),  # Using type_ as type is a reserved keyword
                                        ),
                                        logstore(
                                            chunk(
                                                maxsize("16384"),
                                                compression(
                                                    type_(
                                                        "zstd"
                                                    ),  # Using type_ as type is a reserved keyword
                                                    level("3"),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
            version="1.0",
        )
        generated_xml = generated_services.to_xml()
        # Validate against relaxng
        self.assertTrue(validate_services(etree.fromstring(str(generated_xml))))
        # Check all nodes and attributes being equal
        self.assertTrue(compare_xml(self.xml_schema, str(generated_xml)))


class TestColbertLongServicesConfiguration(unittest.TestCase):
    def setUp(self):
        self.xml_file_path = "tests/testfiles/services/colbert-long/services.xml"
        self.xml_schema = """<?xml version="1.0" encoding="utf-8" ?>

<services version="1.0" xmlns:deploy="vespa" xmlns:preprocess="properties" minimum-required-vespa-version="8.311.28">

    <!-- See https://docs.vespa.ai/en/reference/services-container.html -->
    <container id="default" version="1.0">

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
        <engine>
            <proton>
                <tuning>
                    <searchnode>
                        <requestthreads>
                            <persearch>4</persearch> <!-- Change the number of threads per search here -->
                        </requestthreads>
                    </searchnode>
                </tuning>
            </proton>
        </engine>
    </content>

</services>
"""

    def test_valid_config_from_file(self):
        self.assertTrue(validate_services(self.xml_file_path))

    def test_valid_config_from_string(self):
        self.assertTrue(validate_services(self.xml_schema))

    def test_valid_from_etree(self):
        to_validate = etree.parse(self.xml_file_path)
        self.assertTrue(validate_services(to_validate))

    def test_generate_colbert_long_services(self):
        # Generated XML using dynamic tag functions
        generated_services = services(
            container(id="default", version="1.0")(
                component(id="colbert", type="colbert-embedder")(
                    transformer_model(
                        url="https://huggingface.co/colbert-ir/colbertv2.0/resolve/main/model.onnx"
                    ),
                    tokenizer_model(
                        url="https://huggingface.co/colbert-ir/colbertv2.0/raw/main/tokenizer.json"
                    ),
                ),
                document_api(),
                search(),
                nodes(count="1")(
                    resources(vcpu="4", memory="16Gb", disk="125Gb")(
                        gpu(count="1", memory="16Gb")
                    ),
                ),
            ),
            content(id="text", version="1.0")(
                min_redundancy("2"),
                documents(document(type="doc", mode="index")),
                nodes(count="2"),
                engine(
                    proton(
                        tuning(
                            searchnode(requestthreads(persearch("4"))),
                        ),
                    ),
                ),
            ),
            version="1.0",
            minimum_required_vespa_version="8.311.28",
        )
        generated_xml = generated_services.to_xml()
        # Validate against relaxng
        self.assertTrue(validate_services(etree.fromstring(str(generated_xml))))
        # Check all nodes and attributes being equal
        self.assertTrue(compare_xml(self.xml_schema, str(generated_xml)))


class TestValidateServices(unittest.TestCase):
    def setUp(self):
        # Prepare some sample valid and invalid XML data
        self.valid_xml_content = """<services>
  <container id="music_container" version="1.0">
    <search></search>
    <document-api></document-api>
    <document-processing></document-processing>
  </container>
  <content id="music_content" version="1.0">
    <redundancy>1</redundancy>
    <documents garbage-collection="true">
      <document type="music" mode="index" selection="music.timestamp &gt; now() - 86400"></document>
    </documents>
    <nodes>
      <node distribution-key="0" hostalias="node1"></node>
    </nodes>
  </content>
</services>"""
        self.invalid_xml_content = """<services>
  <container id="music_container" version="1.0">
    <search></search>
    <documents-api></document-api>
    <document-processing></document-processing>
  </container>
  <content id="music_content" version="1.0">
    <redundancy>1</redundancy>
    <documents garbage-collection="true">
      <document type="music" mode="index" selection="music.timestamp &gt; now() - 86400"></document>
    </documents>
    <nodes>
      <node distribution-key="0" hostalias="node1"></node>
    </nodes>
  </content>
</services>"""

        # Create temporary files with valid and invalid XML content
        self.valid_xml_file = "valid_test.xml"
        self.invalid_xml_file = "invalid_test.xml"

        with open(self.valid_xml_file, "w") as f:
            f.write(self.valid_xml_content)

        with open(self.invalid_xml_file, "w") as f:
            f.write(self.invalid_xml_content)

        # Create etree.Element from valid XML content
        self.valid_xml_element = etree.fromstring(self.valid_xml_content)

    def tearDown(self):
        # Clean up temporary files
        os.remove(self.valid_xml_file)
        os.remove(self.invalid_xml_file)

    def test_validate_valid_xml_content(self):
        # Test with valid XML content as string
        result = validate_services(self.valid_xml_content)
        self.assertTrue(result)

    def test_validate_invalid_xml_content(self):
        # Test with invalid XML content as string
        result = validate_services(self.invalid_xml_content)
        self.assertFalse(result)

    def test_validate_valid_xml_file(self):
        # Test with valid XML file path
        result = validate_services(self.valid_xml_file)
        self.assertTrue(result)

    def test_validate_invalid_xml_file(self):
        # Test with invalid XML file path
        result = validate_services(self.invalid_xml_file)
        self.assertFalse(result)

    def test_validate_valid_xml_element(self):
        # Test with valid etree.Element
        result = validate_services(self.valid_xml_element)
        self.assertTrue(result)

    def test_validate_nonexistent_file(self):
        # Test with a non-existent file path
        result = validate_services("nonexistent.xml")
        self.assertFalse(result)

    def test_validate_invalid_input_type(self):
        # Test with invalid input type
        result = validate_services(123)
        self.assertFalse(result)


class TestDocumentExpiry(unittest.TestCase):
    def setUp(self):
        self.xml_schema = """<services>
  <container id="music_container" version="1.0">
    <search></search>
    <document-api></document-api>
    <document-processing></document-processing>
  </container>
  <content id="music_content" version="1.0">
    <redundancy>1</redundancy>
    <documents garbage-collection="true">
      <document type="music" mode="index" selection="music.timestamp &gt; now() - 86400"></document>
    </documents>
    <nodes>
      <node distribution-key="0" hostalias="node1"></node>
    </nodes>
  </content>
</services>
"""

    def test_xml_validation(self):
        self.assertTrue(validate_services(self.xml_schema))

    def test_document_expiry(self):
        application_name = "music"
        generated = services(
            container(
                search(),
                document_api(),
                document_processing(),
                id=f"{application_name}_container",
                version="1.0",
            ),
            content(
                redundancy("1"),
                documents(
                    document(
                        type=application_name,
                        mode="index",
                        selection="music.timestamp > now() - 86400",
                    ),
                    garbage_collection="true",
                ),
                nodes(node(distribution_key="0", hostalias="node1")),
                id=f"{application_name}_content",
                version="1.0",
            ),
        )
        generated_xml = generated.to_xml()
        # Validate against relaxng
        self.assertTrue(validate_services((str(generated_xml))))
        # Compare the generated XML with the schema
        self.assertTrue(compare_xml(self.xml_schema, str(generated_xml)))


class TestUnderscoreAttributes(unittest.TestCase):
    def setUp(self):
        self.xml_schema = """<services version="1.0">
    <container id="colpalidemo_container" version="1.0">
        <search></search>
        <document-api></document-api>
        <document-processing></document-processing>
        <clients>
            <client id="mtls" permissions="read,write">
                <certificate file="security/clients.pem" />
            </client>
            <client id="token_write" permissions="read,write">
                <token id="colpalidemo_write" />
            </client>
            <client id="token_read" permissions="read">
                <token id="colpalidemo_read" />
            </client>
        </clients>
        <config name="container.qr-searchers">
            <tag>
                <bold>
                    <open>&lt;strong&gt;</open>
                    <close>&lt;/strong&gt;</close>
                </bold>
                <separator>...</separator>
            </tag>
        </config>
    </container>
    <content id="colpalidemo_content" version="1.0">
        <redundancy>1</redundancy>
        <documents>
            <document type="pdf_page" mode="index"></document>
        </documents>
        <nodes>
            <node distribution-key="0" hostalias="node1"></node>
        </nodes>
        <config name="vespa.config.search.summary.juniperrc">
            <max_matches>2</max_matches>
            <length>1000</length>
            <surround_max>500</surround_max>
            <min_length>300</min_length>
        </config>
    </content>
</services>
"""

    def test_valid_config_from_string(self):
        self.assertTrue(validate_services(self.xml_schema))

    def test_generate_schema(self):
        generated = services(
            container(
                search(),
                document_api(),
                document_processing(),
                clients(
                    client(
                        certificate(file="security/clients.pem"),
                        id="mtls",
                        permissions="read,write",
                    ),
                    client(
                        token(id="colpalidemo_write"),
                        id="token_write",
                        permissions="read,write",
                    ),
                    client(
                        token(id="colpalidemo_read"),
                        id="token_read",
                        permissions="read",
                    ),
                ),
                config(
                    vt("tag")(
                        vt("bold")(
                            vt("open", "<strong>"),
                            vt("close", "</strong>"),
                        ),
                        vt("separator", "..."),
                    ),
                    name="container.qr-searchers",
                ),
                id="colpalidemo_container",
                version="1.0",
            ),
            content(
                redundancy("1"),
                documents(document(type="pdf_page", mode="index")),
                nodes(node(distribution_key="0", hostalias="node1")),
                config(
                    vt("max_matches", "2", replace_underscores=False),
                    vt("length", "1000"),
                    vt("surround_max", "500", replace_underscores=False),
                    vt("min_length", "300", replace_underscores=False),
                    name="vespa.config.search.summary.juniperrc",
                ),
                id="colpalidemo_content",
                version="1.0",
            ),
            version="1.0",
        )
        generated_xml = generated.to_xml()
        # Validate against relaxng
        self.assertTrue(validate_services(str(generated_xml)))
        self.assertTrue(compare_xml(self.xml_schema, str(generated_xml)))


class TestQueryProfileSimple(unittest.TestCase):
    def setUp(self):
        self.query_profile_xml = """<?xml version="1.0" encoding="utf-8"?>
<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the
project root. -->
<!--
match_avg_top_3_chunk_sim_scores   : 13.383840
match_avg_top_3_chunk_text_scores  : 0.203145
match_bm25(chunks)                 : 0.159914
match_bm25(title)                  : 0.191867
match_max_chunk_sim_scores         : 10.067169
match_max_chunk_text_scores        : 0.153392
Intercept                          : -7.798639
-->
<query-profile id="hybrid">
    <field name="schema">doc</field>
    <field name="ranking.features.query(embedding)">embed(@query)</field>
    <field name="ranking.features.query(float_embedding)">embed(@query)</field>
    <field name="ranking.features.query(intercept)">-7.798639</field>
    <field name="ranking.features.query(avg_top_3_chunk_sim_scores_param)">13.383840</field>
    <field name="ranking.features.query(avg_top_3_chunk_text_scores_param)">0.203145</field>
    <field name="ranking.features.query(bm25_chunks_param)">0.159914</field>
    <field name="ranking.features.query(bm25_title_param)">0.191867</field>
    <field name="ranking.features.query(max_chunk_sim_scores_param)">10.067169</field>
    <field name="ranking.features.query(max_chunk_text_scores_param)">0.153392</field>
    <field name="yql">
        select *
        from %{schema}
        where userInput(@query) or
        ({label:"title_label", targetHits:100}nearestNeighbor(title_embedding, embedding)) or
        ({label:"chunks_label", targetHits:100}nearestNeighbor(chunk_embeddings, embedding))
    </field>
    <field name="hits">10</field>
    <field name="ranking.profile">learned-linear</field>
    <field name="presentation.summary">top_3_chunks</field>
</query-profile>    
"""

    def test_simple_query_profile(self):
        generated = query_profile(
            field("doc", name="schema"),
            field("embed(@query)", name="ranking.features.query(embedding)"),
            field("embed(@query)", name="ranking.features.query(float_embedding)"),
            field("-7.798639", name="ranking.features.query(intercept)"),
            field(
                "13.383840",
                name="ranking.features.query(avg_top_3_chunk_sim_scores_param)",
            ),
            field(
                "0.203145",
                name="ranking.features.query(avg_top_3_chunk_text_scores_param)",
            ),
            field("0.159914", name="ranking.features.query(bm25_chunks_param)"),
            field("0.191867", name="ranking.features.query(bm25_title_param)"),
            field(
                "10.067169", name="ranking.features.query(max_chunk_sim_scores_param)"
            ),
            field(
                "0.153392", name="ranking.features.query(max_chunk_text_scores_param)"
            ),
            field(
                """select *
        from %{schema}
        where userInput(@query) or
        ({label:"title_label", targetHits:100}nearestNeighbor(title_embedding, embedding)) or
        ({label:"chunks_label", targetHits:100}nearestNeighbor(chunk_embeddings, embedding))""",
                name="yql",
            ),
            field("10", name="hits"),
            field("learned-linear", name="ranking.profile"),
            field("top_3_chunks", name="presentation.summary"),
            id="hybrid",
        )
        generated_xml = generated.to_xml()
        # Compare the generated XML with the expected schema
        self.assertTrue(compare_xml(self.query_profile_xml, str(generated_xml)))


class TestQueryProfileVariant(unittest.TestCase):
    def test_query_profile_variant(self):
        # Taken from https://docs.vespa.ai/en/query-profiles.html#variants-and-inheritance
        query_profile_xml = """<query-profile id="multiprofile1"> <!-- A regular profile may define "virtual" children within itself -->

    <!-- Names of the request parameters defining the variant profiles of this. Order matters as described below.
         Each individual value looked up in the profile is resolved from the most specific matching virtual
         variant profile -->
    <dimensions>region,model,bucket</dimensions>

    <!-- Values may be set in the profile itself as usual, this becomes the default values given no matching
         virtual variant provides a value for the property -->
    <field name="a">My general a value</field>

    <!-- The "for" attribute in a child profile supplies values in order for each of the dimensions -->
    <query-profile for="us,nokia,test1">
        <field name="a">My value of the combination us-nokia-test1-a</field>
    </query-profile>

    <!-- Same as [us,*,*]  - trailing "*"'s may be omitted -->
    <query-profile for="us">
        <field name="a">My value of the combination us-a</field>
        <field name="b">My value of the combination us-b</field>
    </query-profile>

    <!-- Given a request which matches both the below, the one which specifies concrete values to the left
         gets precedence over those specifying concrete values to the right
         (i.e the first one gets precedence here) -->
    <query-profile for="us,nokia,*">
        <field name="a">My value of the combination us-nokia-a</field>
        <field name="b">My value of the combination us-nokia-b</field>
    </query-profile>
    <query-profile for="us,*,test1">
        <field name="a">My value of the combination us-test1-a</field>
        <field name="b">My value of the combination us-test1-b</field>
    </query-profile>

</query-profile>"""

        generated = query_profile(
            dimensions("region,model,bucket"),
            field("My general a value", name="a"),
            query_profile(for_="us,nokia,test1")(
                field("My value of the combination us-nokia-test1-a", name="a"),
            ),
            query_profile(for_="us")(
                field("My value of the combination us-a", name="a"),
                field("My value of the combination us-b", name="b"),
            ),
            query_profile(for_="us,nokia,*")(
                field("My value of the combination us-nokia-a", name="a"),
                field("My value of the combination us-nokia-b", name="b"),
            ),
            query_profile(for_="us,*,test1")(
                field("My value of the combination us-test1-a", name="a"),
                field("My value of the combination us-test1-b", name="b"),
            ),
            id="multiprofile1",
        )
        generated_xml = generated.to_xml()
        # Compare the generated XML with the expected schema
        self.assertTrue(compare_xml(query_profile_xml, str(generated_xml)))

    def test_query_profile_multi(self):
        query_profile_xml = """<query-profile id="multi" inherits="default multiDimensions"> <!-- default sets default-index to title -->
  <field name="model"><ref>querybest</ref></field>

  <query-profile for="love,default">
    <field name="model"><ref>querylove</ref></field>
    <field name="model.defaultIndex">default</field>
  </query-profile>

  <query-profile for="*,default">
    <field name="model.defaultIndex">default</field>
  </query-profile>

  <query-profile for="love">
    <field name="model"><ref>querylove</ref></field>
  </query-profile>

  <query-profile for="inheritslove" inherits="rootWithFilter">
    <field name="model.filter">+me</field>
  </query-profile>

</query-profile>"""
        generated = query_profile(
            field(
                ref("querybest"),
                name="model",
            ),
            query_profile(for_="love,default")(
                field(ref("querylove"), name="model"),
                field("default", name="model.defaultIndex"),
            ),
            query_profile(for_="*,default")(
                field("default", name="model.defaultIndex"),
            ),
            query_profile(for_="love")(
                field(ref("querylove"), name="model"),
            ),
            query_profile(for_="inheritslove", inherits="rootWithFilter")(
                field("+me", name="model.filter"),
            ),
            id="multi",
            inherits="default multiDimensions",
        )
        generated_xml = generated.to_xml()
        # Compare the generated XML with the expected schema
        self.assertTrue(compare_xml(query_profile_xml, str(generated_xml)))


class TestQueryProfileTypes(unittest.TestCase):
    def test_query_profile_type_root(self):
        self.query_profile_type_xml = """<query-profile-type id="root" inherits="native">
  <match path="true"/>  
  <field name="indexname" type="string" alias="index-name idx"/>
</query-profile-type>"""
        generated = query_profile_type(
            match_(path="true"),  # Match is sanitized due to python keyword
            field(name="indexname", type="string", alias="index-name idx"),
            id="root",
            inherits="native",
        )
        generated_xml = generated.to_xml()
        # Compare the generated XML with the expected schema
        self.assertTrue(compare_xml(self.query_profile_type_xml, str(generated_xml)))

    def test_query_profile_type_mandatory(self):
        self.query_profile_type_xml = """<query-profile-type id="mandatory" inherits="native">
  <field name="timeout" type="string" mandatory="true"/>
  <field name="foo" type="integer" mandatory="true"/>
</query-profile-type>"""
        generated = query_profile_type(
            field(name="timeout", type="string", mandatory="true"),
            field(name="foo", type="integer", mandatory="true"),
            id="mandatory",
            inherits="native",
        )
        generated_xml = generated.to_xml()
        # Compare the generated XML with the expected schema
        self.assertTrue(compare_xml(self.query_profile_type_xml, str(generated_xml)))


if __name__ == "__main__":
    unittest.main()
