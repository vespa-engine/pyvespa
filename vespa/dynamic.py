from functools import partial


class XMLTag:
    def __init__(self, tag, *children, **attrs):
        self.tag = tag
        self.attrs = attrs
        self.children = list(children)

    def __call__(self, *children):
        self.children.extend(children)
        return self

    def to_xml(self, level=0, indent="  "):
        spaces = indent * level
        attrs = " ".join(f'{k}="{v}"' for k, v in self.attrs.items())
        open_tag = f"{spaces}<{self.tag} {attrs}>" if attrs else f"{spaces}<{self.tag}>"

        if not self.children:
            return f"{open_tag[:-1]} />"

        children_xml = "\n".join(
            child.to_xml(level + 1, indent)
            if isinstance(child, XMLTag)
            else f"{spaces}{indent}{child}"
            for child in self.children
        )
        return f"{open_tag}\n{children_xml}\n{spaces}</{self.tag}>"


def VT(version="1.0", encoding="utf-8"):
    return f'<?xml version="{version}" encoding="{encoding}" ?>'


# Define the tags to be exported
__all__ = list(
    map(
        str.capitalize,
        [
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
            "dispatch DEPRECATED",
            "num-dispatch-groups DEPRECATED",
            "group DEPRECATED",
            "nodeDEPRECATED",
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
            "documentapi",
            "documentprocessing",
            "nodes",
            "jvm",
            "node",
            "search",
            "content",
            "minredundancy",
            "documents",
            "document",
            "component",
            "transformermodel",
            "tokenizermodel",
            "resources",
            "gpu",
            "termscorethreshold",
        ],
    )
) + ["VT"]


# Dynamically create tags using globals()
for tag in __all__:
    globals()[tag] = partial(XMLTag, tag.lower())
