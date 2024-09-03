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


def xml_declaration(version="1.0", encoding="utf-8"):
    return f'<?xml version="{version}" encoding="{encoding}" ?>'


# Define the tags to be exported
__all__ = [
    "Services",
    "Admin",
    "Configservers",
    "Configserver",
    "ClusterControllers",
    "ClusterController",
    "Slobroks",
    "Slobrok",
    "AdminServer",
    "Container",
    "DocumentApi",
    "DocumentProcessing",
    "Nodes",
    "Jvm",
    "Node",
    "Search",
    "Content",
    "MinRedundancy",
    "Documents",
    "Document",
    "Component",
    "TransformerModel",
    "TokenizerModel",
    "Resources",
    "Gpu",
    "TermScoreThreshold",
    "xml_declaration",
]


# Dynamically create tags using globals()
for tag in __all__:
    globals()[tag] = partial(XMLTag, tag.lower())
