"""Utilities to express hosts.xml using the VT (Vespa Tag) syntax, which
maps python functions directly to xml tags. This allows you to express any
hosts.xml-configuration in python.

This mirrors the approach used for deployment configuration in
`vespa.configuration.deployment` and allows users to build a hosts
descriptor directly in Python, ensuring the top-level tag is `hosts`.

Example:

    from vespa.configuration.hosts import hosts, host, alias
    from vespa.package import ApplicationPackage

    hosts_cfg = hosts(
        host(name="localhost")(
            alias("node1"),
        ),
        host(name="node2.example.com")(
            alias("node2"),
        ),
    )

    app_package = ApplicationPackage(name="myapp", hosts_config=hosts_cfg)
    # app_package.hosts_to_text now returns the expected hosts.xml content
    # and hosts.xml is included when app_package.to_files() or to_zip() is called.
"""

from typing import List
from vespa.configuration.vt import VT, create_tag_function, voids

# Tags used (subset guided by reference examples). Add more as needed.
# NB! If you modify this, please run 'python vespa/configuration/generate_pyi.py' to regenerate the .pyi stub files.
hosts_tags: List[str] = [
    "hosts",
    "host",
    "alias",
]

# Fail fast on duplicates
_dup = {t for t in hosts_tags if hosts_tags.count(t) > 1}
if _dup:
    raise ValueError(f"Tags duplicated in hosts_tags: {_dup}")

generated_hosts_tags = set()
g = globals()
for _tag in hosts_tags:
    sanitized_name = VT.sanitize_tag_name(_tag)
    generated_hosts_tags.add(sanitized_name)
    g[sanitized_name] = create_tag_function(_tag, _tag in voids)


class HostsItem:
    """Wrapper around a VT structure representing a full hosts.xml tree.

    Ensures that the provided VT has a `hosts` top-level tag and exposes
    the rendered XML via the `to_xml` method.
    """

    def __init__(self, root: VT):
        if not isinstance(root, VT):  # defensive
            raise TypeError(
                f"HostsItem expects a VT root element, got {type(root).__name__}"
            )
        if root.tag != VT.sanitize_tag_name("hosts"):
            raise ValueError("Top level tag for hosts configuration must be 'hosts'")
        self.root = root

    @classmethod
    def from_vt(cls, vt_obj: VT) -> "HostsItem":
        return cls(vt_obj)

    def to_xml(self) -> str:
        return str(self.root.to_xml()).strip()

    def __repr__(self) -> str:  # pragma: no cover
        return f"HostsItem(root={self.root!r})"
