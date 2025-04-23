from fastcore.utils import tuplify
import types
from xml.sax.saxutils import escape
from fastcore.utils import patch
import xml.etree.ElementTree as ET

# If the vespa tags correspond to reserved Python keywords or commonly used names,
# they are replaced with the following:
replace_reserved = {
    "type": "type_",
    "class": "class_",
    "for": "for_",
    "time": "time_",
    "io": "io_",
    "from": "from_",
}
restore_reserved = {v: k for k, v in replace_reserved.items()}


class VT:
    "A 'Vespa Tag' structure, containing `tag`, `children`, and `attrs`"

    @staticmethod
    def sanitize_tag_name(tag: str) -> str:
        "Convert invalid tag names (with '-') to valid Python identifiers (with '_')"
        replaced = tag.replace("-", "_")
        return replace_reserved.get(replaced, replaced)

    def __init__(
        self,
        tag: str,
        cs: tuple,
        attrs: dict = None,
        void_=False,
        replace_underscores: bool = True,
        **kwargs,
    ):
        assert isinstance(cs, tuple)
        self.tag = self.sanitize_tag_name(tag)  # Sanitize tag name
        self.children, self.attrs = cs, attrs or {}
        self.void_ = void_
        self.replace_underscores = replace_underscores

    def __setattr__(self, k, v):
        if k.startswith("__") or k in (
            "tag",
            "children",
            "attrs",
            "void_",
            "replace_underscores",
        ):
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

    def restore_tag_name(
        self,
    ) -> str:
        "Restore sanitized tag names back to the original names for XML generation"
        restored = restore_reserved.get(self.tag, self.tag)
        if self.replace_underscores:
            return restored.replace("_", "-")
        return restored

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
    """This maps the attributes that we don't want to be Python keywords or commonly used names to the replacement names."""
    o = dict(_global="global").get(o, o)
    return o.lstrip("_").replace("_", "-")


def valmap(o):
    """Convert values to the string representation for xml. integers to strings and booleans to 'true' or 'false'"""
    if isinstance(o, bool):
        return str(o).lower()
    elif isinstance(o, int):
        return str(o)
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
    """
    Preprocess the children and attributes of a VT structure.

    :param c: Children of the VT structure
    :param kw: Attributes of the VT structure
    :param attrmap: Dict to map attribute names
    :param valmap: Dict to map attribute values

    :return: Tuple of children and attributes
    """

    # If the children are a single generator, map, or filter, convert it to a tuple
    if len(c) == 1 and isinstance(c[0], (types.GeneratorType, map, filter)):
        c = tuple(c[0])
    # Create the attributes dictionary by mapping the keys and values
    # TODO: Check if any of Vespa supported attributes are camelCase
    attrs = {attrmap(k.lower()): valmap(v) for k, v in kw.items() if v is not None}
    return _flatten_tuple(c), attrs


# General XML tag creation function
def vt(
    tag: str,
    *c,
    void_: bool = False,
    attrmap: callable = attrmap,
    valmap: callable = valmap,
    replace_underscores: bool = True,
    **kw,
):
    "Create a VT structure with `tag`, `children` and `attrs`"
    # NB! fastcore.xml uses tag.lower() for tag names. This is not done here.
    return VT(
        tag,
        *_preproc(c, kw, attrmap=attrmap, valmap=valmap),
        void_=void_,
        replace_underscores=replace_underscores,
    )


# XML void tags (self-closing)
# TODO: Add self-closing tags for Vespa configuration
voids = set("".split())


def Xml(*c, version="1.0", encoding="UTF-8", **kwargs) -> VT:
    "A top level XML tag, with `encoding` and children `c`"
    res = vt("?xml", *c, version=version, encoding=encoding, void_="?")
    return res


# Replace the 'partial' based tag creation
def create_tag_function(tag, void_):
    def tag_function(*c, **kwargs):
        return vt(tag, *c, void_=void_, **kwargs)

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
        # Ensure text content is compact (no trailing newline unless indent=True)
        return f"{esc_fn(str(elm).strip())}{nl if indent else ''}"

    tag, cs, attrs = elm.list

    stag = elm.restore_tag_name()

    # Prepare the attribute string only once
    attr_str = ""
    if attrs:
        attr_str = " " + " ".join(_to_attr(k, v) for k, v in attrs.items())

    # Handle void (self-closing) tags
    if elm.void_:
        if isinstance(elm.void_, str):
            return f"{sp}<{stag}{attr_str} {elm.void_}>{nl}"
        return f"{sp}<{stag}{attr_str} />{nl}"

    # Handle non-void tags with children or no children
    if cs:
        # Handle the case where children are text or elements
        res = f"{sp}<{stag}{attr_str}>"

        # If the children are just text or int, don't introduce newlines
        if len(cs) == 1 and (isinstance(cs[0], str) or isinstance(cs[0], int)):
            if isinstance(cs[0], int):
                cs = str(cs[0])
            res += f"{esc_fn(cs[0].strip())}</{stag}>{nl if indent else ''}"
        else:
            # If there are multiple children, properly indent them
            res += f"{nl if indent else ''}"
            res += "".join(
                _to_xml(c, lvl=lvl + 2, indent=indent, do_escape=do_escape) for c in cs
            )
            res += f"{sp}</{stag}>{nl if indent else ''}"

        return Safe(res)
    else:
        # Non-void tag without children
        return f"{sp}<{stag}{attr_str}></{stag}>{nl if indent else ''}"


def to_xml(elm, lvl=0, indent: bool = True, do_escape: bool = True):
    "Convert `vt` element tree into an XML string"
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


def canonicalize(element):
    """Recursively sort attributes and children to canonicalize the element."""
    # Sort attributes
    if element.attrib:
        element.attrib = dict(sorted(element.attrib.items()))
    # Sort children by tag and text
    children = list(element)
    for child in children:
        canonicalize(child)
    element[:] = sorted(children, key=lambda e: (e.tag, (e.text or "").strip()))
    # Strip whitespace from text and tail
    if element.text:
        element.text = element.text.strip()
    if element.tail:
        element.tail = element.tail.strip()


def elements_equal(e1, e2):
    """Compare two elements for equality."""
    if e1.tag != e2.tag:
        return False
    if sorted(e1.attrib.items()) != sorted(e2.attrib.items()):
        return False
    if (e1.text or "").strip() != (e2.text or "").strip():
        return False
    if (e1.tail or "").strip() != (e2.tail or "").strip():
        return False
    if len(e1) != len(e2):
        return False
    return all(elements_equal(c1, c2) for c1, c2 in zip(e1, e2))


def compare_xml(xml_str1, xml_str2):
    """Compare two XML strings for equality."""
    try:
        tree1 = ET.ElementTree(ET.fromstring(xml_str1))
        tree2 = ET.ElementTree(ET.fromstring(xml_str2))
    except ET.ParseError:
        return False
    root1 = tree1.getroot()
    root2 = tree2.getroot()
    canonicalize(root1)
    canonicalize(root2)
    return elements_equal(root1, root2)
