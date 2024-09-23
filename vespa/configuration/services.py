from vespa.configuration.vt import VT, create_tag_function, voids
from lxml import etree
from typing import Union
import pathlib
import os

project_dir = pathlib.Path(__file__).parent.parent.parent
# Load the RelaxNG schema at the module level to avoid reloading it every time
try:
    with open(
        project_dir / "tests/testfiles/relaxng/services.rng", "rb"
    ) as schema_file:
        RELAXNG_SCHEMA = etree.RelaxNG(etree.parse(schema_file))
except (OSError, etree.XMLSyntaxError) as e:
    print(f"Error loading RelaxNG schema: {e}")
    RELAXNG_SCHEMA = None

# List of XML tags (customized for Vespa configuration)
services_tags = [
    "redundancy",
    "documents",
    "merges",
    "config",
    "max-hits-per-partition",
    "query",
    "flushstrategy",
    "memory",
    "chain",
    "dispatch-policy",
    "conservative",
    "top-k-probability",
    "cache",
    "filtering",
    "maxmemorygain",
    "components",
    "persistence-threads",
    "server",
    "transformer-model",
    "transition-time",
    "secret-store",
    "document-processing",
    "documentprocessor",
    "maintenance",
    "slobroks",
    "clustercontrollers",
    "gpu-device",
    "resource-limits",
    "summary",
    "visibility-delay",
    "timeout",
    "gpu",
    "cluster-controller",
    "level",
    "min-active-docs-coverage",
    "renderer",
    "lidspace",
    "memory-limit-factor",
    "slobrok",
    "container",
    "abortondocumenterror",
    "search",
    "bucket-splitting",
    "execution-mode",
    "diskbloatfactor",
    "jvm",
    "environment-variables",
    "federation",
    "node",
    "retryenabled",
    "tokenizer-model",
    "http",
    "maxpendingbytes",
    "maxpendingdocs",
    "visitors",
    "zookeeper",
    "unpack",
    "read",
    "max-concurrent",
    "clustercontroller",
    "onnx",
    "searcher",
    "native",
    "proton",
    "total",
    "flush-on-shutdown",
    "warmup",
    "document-api",
    "coverage",
    "age",
    "retrydelay",
    "feeding",
    "services",
    "distribution",
    "min-wait-after-coverage-factor",
    "initialize",
    "disk-limit-factor",
    "type",
    "stable-state-period",
    "persearch",
    "mbusport",
    "ignore-undefined-fields",
    "groups-allowed-down-ratio",
    "io",
    "processor",
    "searchable-copies",
    "route",
    "interval",
    "min-distributor-up-ratio",
    "searchnode",
    "max-bloat-factor",
    "binding",
    "interop-threads",
    "tuning",
    "maxsize-percent",
    "admin",
    "init-progress-time",
    "adminserver",
    "dispatch",
    "threadpool",
    "maxage",
    "accesslog",
    "handler",
    "tracelevel",
    "logstore",
    "intraop-threads",
    "maxfilesize",
    "processing",
    "model",
    "chunk",
    "minimum",
    "configservers",
    "provider",
    "min-storage-up-ratio",
    "prepend",
    "min-redundancy",
    "models",
    "query-timeout",
    "index",
    "group",
    "include",
    "requestthreads",
    "nodes",
    "disk",
    "time",
    "engine",
    "prune",
    "resources",
    "compression",
    "content",
    "concurrency",
    "niceness",
    "sync-transactionlog",
    "transactionlog",
    "threads",
    "max-premature-crashes",
    "configserver",
    "model-evaluation",
    "component",
    "document",
    "store",
    "min-node-ratio-per-group",
    "removed-db",
    "max-wait-after-coverage-factor",
    "maxsize",
    "clients",
    "client",
]

# Fail if any tag is duplicated. Provide feedback of which tags are duplicated.
duplicated = set([tag for tag in services_tags if services_tags.count(tag) > 1])
if duplicated:
    raise ValueError(f"Tags duplicated in services_tags: {duplicated}")
# Mapping general XML tags
_g = globals()

# Generate dynamic tag functions and map them to valid Python names
for tag in services_tags:
    sanitized_name = VT.sanitize_tag_name(tag)
    _g[sanitized_name] = create_tag_function(tag, tag in voids)


def validate_services(
    xml_input: Union[str, pathlib.Path, etree._ElementTree, etree._Element],
) -> bool:
    """
    Validate an XML input against the RelaxNG schema file for services.xml

    Args:
        xml_input: XML input to validate. It can be:
            - A string containing XML data
            - A string or pathlib.Path pointing to an XML file
            - An etree._Element or etree._ElementTree instance

    Returns:
        True if the XML input is valid according to the RelaxNG schema, False otherwise.
    """
    if RELAXNG_SCHEMA is None:
        print("RelaxNG schema is not loaded.")
        return False

    try:
        # If xml_input is an etree Element or ElementTree, use it directly
        if isinstance(xml_input, (etree._ElementTree, etree._Element)):
            xml_doc = xml_input
        # If it's a Path object or a string that points to an existing file, parse it from the file
        elif isinstance(xml_input, (str, pathlib.Path)) and os.path.exists(
            str(xml_input)
        ):
            xml_doc = etree.parse(str(xml_input))
        # Otherwise, try to parse it as an XML string
        elif isinstance(xml_input, str):
            xml_doc = etree.fromstring(xml_input.encode("utf-8"))
        else:
            print("Invalid input type for xml_input.")
            return False

        # Validate the XML document against the RelaxNG schema
        is_valid = RELAXNG_SCHEMA.validate(xml_doc)
        if not is_valid:
            print(f"Validation errors:\n{RELAXNG_SCHEMA.error_log}")
        return is_valid

    except (etree.XMLSyntaxError, OSError) as e:
        print(f"Error parsing XML input: {e}")
        return False
