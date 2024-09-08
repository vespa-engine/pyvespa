from fastcore.utils import tuplify
import types
from xml.sax.saxutils import escape
from fastcore.utils import patch


# %% ../nbs/11_xml.ipynb
class FT:
    "A 'Fast Tag' structure, containing `tag`, `children`, and `attrs`"

    @staticmethod
    def sanitize_tag_name(tag: str) -> str:
        "Convert invalid tag names (with '-') to valid Python identifiers (with '_')"
        return tag.replace("-", "_")

    @staticmethod
    def restore_tag_name(tag: str) -> str:
        "Restore sanitized tag names back to the original names for XML generation"
        return tag.replace("_", "-")

    def __init__(self, tag: str, cs: tuple, attrs: dict = None, void_=False, **kwargs):
        assert isinstance(cs, tuple)
        self.tag = self.sanitize_tag_name(tag)  # Sanitize tag name
        self.children, self.attrs = cs, attrs or {}
        self.void_ = void_

    def __setattr__(self, k, v):
        if k.startswith("__") or k in ("tag", "children", "attrs", "void_"):
            return super().__setattr__(k, v)
        self.attrs[k.lstrip("_").replace("_", "-")] = v

    def __getattr__(self, k):
        if k.startswith("__"):
            raise AttributeError(k)
        return self.get(k)

    @property
    def list(self):
        return [self.tag, self.children, self.attrs]

    def get(self, k, default=None):
        return self.attrs.get(k.lstrip("_").replace("_", "-"), default)

    def __repr__(self):
        return f"{self.tag}({self.children},{self.attrs})"

    def __add__(self, b):
        self.children = self.children + tuplify(b)
        return self

    def __getitem__(self, idx):
        return self.children[idx]

    def __iter__(self):
        return iter(self.children)


# List of XML tags (customized for Vespa configuration)
xml_tags = list(
    map(
        FT.sanitize_tag_name,
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
# %% auto 0
__all__ = [
    "FT",
    "attrmap",
    "valmap",
    "ft",
    "Safe",
    "to_xml",
    "highlight",
    "showtags",
] + xml_tags

# %% ../nbs/11_xml.ipynb


# %% ../nbs/11_xml.ipynb
def attrmap(o):
    return o.lstrip("_").replace("_", "-")


def valmap(o):
    return o if isinstance(o, str) else " ".join(map(str, o))


def _flatten_tuple(tup):
    result = []
    for item in tup:
        if isinstance(item, tuple):
            result.extend(item)
        else:
            result.append(item)
    return tuple(result)


def _preproc(c, kw, attrmap=attrmap, valmap=valmap):
    if len(c) == 1 and isinstance(c[0], (types.GeneratorType, map, filter)):
        c = tuple(c[0])
    attrs = {attrmap(k.lower()): valmap(v) for k, v in kw.items() if v is not None}
    return _flatten_tuple(c), attrs


# General XML tag creation function
def ft(
    tag: str,
    *c,
    void_: bool = False,
    attrmap: callable = attrmap,
    valmap: callable = valmap,
    **kw,
):
    "Create an `FT` structure for `to_xml()`"
    return FT(
        tag.lower(), *_preproc(c, kw, attrmap=attrmap, valmap=valmap), void_=void_
    )


# XML void tags (self-closing)
voids = set(
    "area base br col command embed hr img input keygen link meta param source track wbr".split()
)

# Mapping general XML tags
_g = globals()


# Replace the 'partial' based tag creation
def create_tag_function(tag, void_):
    def tag_function(*c, **kwargs):
        return ft(tag, *c, void_=void_, **kwargs)

    tag_function.__name__ = FT.sanitize_tag_name(tag)  # Assigning sanitized tag name
    return tag_function


# Generate dynamic tag functions and map them to valid Python names
for tag in xml_tags:
    sanitized_name = FT.sanitize_tag_name(tag)
    _g[sanitized_name] = create_tag_function(tag, tag in voids)


# %% ../nbs/11_xml.ipynb
def _escape(s):
    return escape(s) if isinstance(s, str) else s


def _to_attr(k, v):
    if isinstance(v, bool):
        return f"{k}" if v else ""
    return f'{k}="{escape(str(v))}"'


def _to_xml(elm, lvl, indent, do_escape):
    esc_fn = _escape if do_escape else lambda s: s
    nl = "\n"
    sp = " " * lvl if indent else ""

    if elm is None:
        return ""

    if isinstance(elm, tuple):
        return (
            f"{nl}".join(
                _to_xml(o, lvl=lvl, indent=indent, do_escape=do_escape) for o in elm
            )
            + nl
        )

    if not isinstance(elm, FT):
        return f"{esc_fn(elm)}{nl}"

    tag, cs, attrs = elm.list
    stag = FT.restore_tag_name(tag)

    # Prepare the attribute string only once
    attr_str = ""
    if attrs:
        attr_str = " " + " ".join(_to_attr(k, v) for k, v in attrs.items())

    # Handle void (self-closing) tags
    if elm.void_:
        return f"{sp}<{stag}{attr_str} />{nl}"

    # Handle non-void tags with children or no children
    if cs:
        res = f"{sp}<{stag}{attr_str}>{nl if indent else ''}"
        res += "".join(
            _to_xml(c, lvl=lvl + 2, indent=indent, do_escape=do_escape) for c in cs
        )
        res += f"{sp}</{stag}>{nl}"
        return Safe(res)
    else:
        return f"{sp}<{stag}{attr_str}></{stag}>{nl}"


def to_xml(elm, lvl=0, indent: bool = True, do_escape: bool = True):
    "Convert `ft` element tree into an XML string"
    return Safe(_to_xml(elm, lvl, indent, do_escape=do_escape))


FT.__html__ = to_xml


# %% ../nbs/11_xml.ipynb
class Safe(str):
    def __html__(self):
        return self


# %% ../nbs/11_xml.ipynb
def highlight(s, lang="xml"):
    "Markdown to syntax-highlight `s` in language `lang`"
    return f"```{lang}\n{to_xml(s)}\n```"


# %% ../nbs/11_xml.ipynb
def showtags(s):
    return f"""<code><pre>{escape(to_xml(s))}</code></pre>"""


FT._repr_markdown_ = highlight


# %% ../nbs/11_xml.ipynb
@patch
def __call__(self: FT, *c, **kw):
    c, kw = _preproc(c, kw)
    if c:
        self = self + c
    if kw:
        self.attrs = {**self.attrs, **kw}
    return self
