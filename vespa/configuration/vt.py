from fastcore.utils import tuplify
import types
from xml.sax.saxutils import escape
from fastcore.utils import patch


class VT:
    "A 'Vespa Tag' structure, containing `tag`, `children`, and `attrs`"

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
    "Create an `VT` structure for `to_xml()`"
    return VT(
        tag.lower(), *_preproc(c, kw, attrmap=attrmap, valmap=valmap), void_=void_
    )


# XML void tags (self-closing)
# TODO: Add self-closing tags for Vespa configuration
voids = set("".split())


# Replace the 'partial' based tag creation
def create_tag_function(tag, void_):
    def tag_function(*c, **kwargs):
        return ft(tag, *c, void_=void_, **kwargs)

    tag_function.__name__ = VT.sanitize_tag_name(tag)  # Assigning sanitized tag name
    return tag_function


def vt_escape(s):
    return escape(s) if isinstance(s, str) else s


def _to_attr(k, v):
    if isinstance(v, bool):
        return f"{k}" if v else ""
    return f'{k}="{escape(str(v))}"'


def _to_xml(elm, lvl, indent, do_escape):
    esc_fn = vt_escape if do_escape else lambda s: s
    nl = "\n" if indent else ""
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

    if not isinstance(elm, VT):
        return f"{esc_fn(elm)}{nl}"

    tag, cs, attrs = elm.list
    stag = VT.restore_tag_name(tag)

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
        res += f"{sp}</{stag}>{nl if indent else ''}"
        return Safe(res)
    else:
        # This is where we remove the newline in the closing tag if indent=False
        return f"{sp}<{stag}{attr_str}></{stag}>{nl if indent else ''}"


def to_xml(elm, lvl=0, indent: bool = True, do_escape: bool = True):
    "Convert `ft` element tree into an XML string"
    return Safe(_to_xml(elm, lvl, indent, do_escape=do_escape))


VT.to_xml = to_xml


class Safe(str):
    def __html__(self):
        return self


def highlight(s, lang="xml"):
    "Markdown to syntax-highlight `s` in language `lang`"
    return f"```{lang}\n{to_xml(s)}\n```"


def showtags(s):
    return f"""<code><pre>{escape(to_xml(s))}</code></pre>"""


VT._repr_markdown_ = highlight


@patch
def __call__(self: VT, *c, **kw):
    c, kw = _preproc(c, kw)
    if c:
        self = self + c
    if kw:
        self.attrs = {**self.attrs, **kw}
    return self
