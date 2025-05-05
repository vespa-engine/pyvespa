from vespa.configuration.vt import VT, create_tag_function, voids
from vespa.configuration.relaxng import RELAXNG
from lxml import etree
from pathlib import Path
import os
from typing import Union

# List of XML tags (customized for Vespa configuration)
# Please keep this in alphabetical order so it is easier to maintain.
# The list is used to generate functions for each tag dynamically.
services_tags = [
    "abortondocumenterror",
    "accesslog",
    "admin",
    "adminserver",
    "age",
    "binding",
    "bucket-splitting",
    "cache",
    "certificate",
    "chain",
    "chunk",
    "client",
    "clients",
    "cluster-controller",
    "clustercontroller",
    "clustercontrollers",
    "component",
    "components",
    "compression",
    "concurrency",
    "config",
    "configserver",
    "configservers",
    "conservative",
    "container",
    "content",
    "coverage",
    "disk",
    "disk-limit-factor",
    "diskbloatfactor",
    "dispatch",
    "dispatch-policy",
    "distribution",
    "document",
    "document-api",
    "document-processing",
    "document-token-id",
    "documentprocessor",
    "documents",
    "engine",
    "environment-variables",
    "execution-mode",
    "federation",
    "feeding",
    "filtering",
    "flush-on-shutdown",
    "flushstrategy",
    "gpu",
    "gpu-device",
    "group",
    "groups-allowed-down-ratio",
    "handler",
    "http",
    "ignore-undefined-fields",
    "include",
    "index",
    "init-progress-time",
    "initialize",
    "interop-threads",
    "interval",
    "intraop-threads",
    "io",
    "jvm",
    "level",
    "lidspace",
    "logstore",
    "maintenance",
    "max-bloat-factor",
    "max-concurrent",
    "max-document-tokens",
    "max-hits-per-partition",
    "max-premature-crashes",
    "max-query-tokens",
    "max-tokens",
    "max-wait-after-coverage-factor",
    "maxage",
    "maxfilesize",
    "maxmemorygain",
    "maxpendingbytes",
    "maxpendingdocs",
    "maxsize",
    "maxsize-percent",
    "mbusport",
    "memory",
    "memory-limit-factor",
    "merges",
    "min-active-docs-coverage",
    "min-distributor-up-ratio",
    "min-node-ratio-per-group",
    "min-redundancy",
    "min-storage-up-ratio",
    "min-wait-after-coverage-factor",
    "minimum",
    "model",
    "model-evaluation",
    "models",
    "native",
    "niceness",
    "node",
    "nodes",
    "onnx",
    "onnx-execution-mode",
    "onnx-gpu-device",
    "onnx-interop-threads",
    "onnx-intraop-threads",
    "persearch",
    "persistence-threads",
    "pooling-strategy",
    "prepend",
    "processing",
    "processor",
    "proton",
    "provider",
    "prune",
    "query",
    "query-timeout",
    "query-token-id",
    "read",
    "redundancy",
    "removed-db",
    "renderer",
    "requestthreads",
    "resource-limits",
    "resources",
    "retrydelay",
    "retryenabled",
    "route",
    "search",
    "searchable-copies",
    "searcher",
    "searchnode",
    "secret-store",
    "server",
    "services",
    "slobrok",
    "slobroks",
    "stable-state-period",
    "store",
    "summary",
    "sync-transactionlog",
    "term-score-threshold",
    "threadpool",
    "threads",
    "time",
    "timeout",
    "token",
    "tokenizer-model",
    "top-k-probability",
    "total",
    "tracelevel",
    "transactionlog",
    "transformer-attention-mask",
    "transformer-end-sequence-token",
    "transformer-input-ids",
    "transformer-mask-token",
    "transformer-model",
    "transformer-output",
    "transformer-pad-token",
    "transformer-start-sequence-token",
    "transition-time",
    "tuning",
    "type",
    "unpack",
    "visibility-delay",
    "visitors",
    "warmup",
    "zookeeper",
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


def validate_services(xml_input: Union[Path, str, etree.Element]) -> bool:
    """
    Validate an XML input against the RelaxNG schema file for services.xml

    Args:
        xml_input (Path or str or etree.Element): The XML input to validate.
    Returns:
        True if the XML input is valid according to the RelaxNG schema, False otherwise.
    """
    try:
        if isinstance(xml_input, etree._Element):
            xml_tree = etree.ElementTree(xml_input)
        elif isinstance(xml_input, etree._ElementTree):
            xml_tree = xml_input
        elif isinstance(xml_input, (str, Path)):
            if isinstance(xml_input, Path) or os.path.exists(xml_input):
                # Assume it's a file path
                xml_tree = etree.parse(str(xml_input))
            elif isinstance(xml_input, str):
                # May hav unicode string with encoding declaration
                if "encoding" in xml_input:
                    xml_tree = etree.ElementTree(etree.fromstring(xml_input.encode()))
                else:
                    # Assume it's a string containing XML content
                    xml_tree = etree.ElementTree(etree.fromstring(xml_input))
        else:
            raise TypeError("xml_input must be a Path, str, or etree.Element.")
    except Exception as e:
        # Handle parsing exceptions
        print(f"Error parsing XML input: {e}")
        return False

    is_valid = RELAXNG["services"].validate(xml_tree)
    return is_valid
