from dynamic import *

# Construct the XML structure
xml_structure = Services(
    version="1.0", xmlns_deploy="vespa", xmlns_preprocess="properties"
)(
    Admin(version="2.0")(
        Configservers(
            Configserver(hostalias="node0"),
            Configserver(hostalias="node1"),
            Configserver(hostalias="node2"),
        ),
        Clustercontrollers(
            Clustercontroller(hostalias="node0", jvm_options="-Xms32M -Xmx64M"),
            Clustercontroller(hostalias="node1", jvm_options="-Xms32M -Xmx64M"),
            Clustercontroller(hostalias="node2", jvm_options="-Xms32M -Xmx64M"),
        ),
        Slobroks(
            Slobrok(hostalias="node0"),
            Slobrok(hostalias="node1"),
            Slobrok(hostalias="node2"),
        ),
        Adminserver(hostalias="node3"),
    ),
    Container(id="feed", version="1.0")(
        Documentapi(),
        Documentprocessing(),
        Nodes(
            Jvm(options="-Xms32M -Xmx128M"),
            Node(hostalias="node4"),
            Node(hostalias="node5"),
        ),
    ),
    Container(id="query", version="1.0")(
        Search(),
        Nodes(
            Jvm(options="-Xms32M -Xmx128M"),
            Node(hostalias="node6"),
            Node(hostalias="node7"),
        ),
    ),
    Content(id="music", version="1.0")(
        Minredundancy("2"),
        Documents(
            Document(type="music", mode="index"),
            Documentprocessing(cluster="feed"),
        ),
        Nodes(
            Node(hostalias="node8", distribution_key="0"),
            Node(hostalias="node9", distribution_key="1"),
        ),
    ),
)

# Add XML declaration and print the XML string
xml_output = f"{VT()}\n{xml_structure.to_xml()}"
print(xml_output)

# Construct the XML structure
xml_structure = Services(
    version="1.0", xmlns_deploy="vespa", xmlns_preprocess="properties"
)(
    Container(id="default", version="1.0")(
        Component(id="splade", type="splade-embedder")(
            Transformermodel(
                url="https://huggingface.co/Qdrant/Splade_PP_en_v1/resolve/main/model.onnx"
            ),
            Tokenizermodel(
                url="https://huggingface.co/Qdrant/Splade_PP_en_v1/raw/main/tokenizer.json"
            ),
            Termscorethreshold("0.8"),
        ),
        Documentapi(),
        Search(),
        Nodes(count="1")(
            Resources(vcpu="4", memory="16Gb", disk="125Gb")(
                Gpu(count="1", memory="16Gb")
            )
        ),
    ),
    Content(id="text", version="1.0")(
        Minredundancy("2"),
        Documents(Document(type="doc", mode="index")),
        Nodes(count="2"),
    ),
)

# Add XML declaration and print the XML string
xml_output = f"{VT()}\n{xml_structure.to_xml()}"
print(xml_output)
