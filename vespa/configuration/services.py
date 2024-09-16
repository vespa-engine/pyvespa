from vespa.configuration.vt import VT, create_tag_function, voids
from lxml import etree

# List of XML tags (customized for Vespa configuration)
services_tags = list(
    map(
        VT.sanitize_tag_name,
        [
            "services",
            "content",
            "documents",
            "document",
            "document-processing",
            "min-redundancy",
            "redundancy",
            "nodes",
            "node",
            "group",
            "distribution",
            "node",
            "group",
            "engine",
            "proton",
            "searchable-copies",
            "tuning",
            "searchnode",
            "lidspace",
            "max-bloat-factor",
            "requestthreads",
            "search",
            "persearch",
            "summary",
            "flushstrategy",
            "native",
            "total",
            "maxmemorygain",
            "diskbloatfactor",
            "component",
            "maxmemorygain",
            "diskbloatfactor",
            "maxage",
            "transactionlog",
            "maxsize",
            "conservative",
            "memory-limit-factor",
            "disk-limit-factor",
            "initialize",
            "threads",
            "feeding",
            "concurrency",
            "niceness",
            "index",
            "io",
            "search",
            "warmup",
            "time",
            "unpack",
            "removed-db",
            "prune",
            "age",
            "interval",
            "summary",
            "io",
            "read",
            "store",
            "cache",
            "maxsize",
            "maxsize-percent",
            "compression",
            "type",
            "level",
            "logstore",
            "maxfilesize",
            "chunk",
            "maxsize",
            "compression",
            "type",
            "level",
            "sync-transactionlog",
            "flush-on-shutdown",
            "resource-limits",
            "disk",
            "memory",
            "search",
            "query-timeout",
            "visibility-delay",
            "coverage",
            "minimum",
            "min-wait-after-coverage-factor",
            "max-wait-after-coverage-factor",
            "tuning",
            "bucket-splitting",
            "min-node-ratio-per-group",
            "distribution",
            "maintenance",
            "merges",
            "persistence-threads",
            "resource-limits",
            "visitors",
            "max-concurrent",
            "dispatch",
            "max-hits-per-partition",
            "top-k-probability",
            "dispatch-policy",
            "min-active-docs-coverage",
            "cluster-controller",
            "init-progress-time",
            "transition-time",
            "max-premature-crashes",
            "stable-state-period",
            "min-distributor-up-ratio",
            "min-storage-up-ratio",
            "groups-allowed-down-ratio",
            "services",
            "admin",
            "configservers",
            "configserver",
            "clustercontrollers",
            "clustercontroller",
            "slobroks",
            "slobrok",
            "adminserver",
            "container",
            "document-api",
            "nodes",
            "jvm",
            "node",
            "search",
            "content",
            "min-redundancy",
            "component",
            "transformer-model",
            "tokenizer-model",
            "prepend",
            "query",
            "resources",
            "gpu",
        ],
    )
)
# Mapping general XML tags
_g = globals()

# Generate dynamic tag functions and map them to valid Python names
for tag in services_tags:
    sanitized_name = VT.sanitize_tag_name(tag)
    _g[sanitized_name] = create_tag_function(tag, tag in voids)

with open("tests/testfiles/relaxng/services.rng", "rb") as schema_file:
    relaxng = etree.RelaxNG(etree.parse(schema_file))


def validate_services(xml_schema: str) -> bool:
    """
    Validate an XML schema against the RelaxNG schema file for services.xml

    Args:
        xml_schema (str): XML schema to validate

    Returns:
        bool: True if the XML schema is valid, False otherwise
    """
    return relaxng.validate(xml_schema)
