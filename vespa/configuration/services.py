from vespa.configuration.vt import VT, create_tag_function, voids
from vespa.configuration.relaxng import RELAXNG
from lxml import etree
from pathlib import Path
import os
from typing import Union

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
    "certificate",
    "token",
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
