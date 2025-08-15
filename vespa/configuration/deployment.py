"""Utilities to express deployment.xml using the VT (Vespa Tag) syntax, which
maps python functions directly to xml tags. This allows you to express any deployment.xml-configuration
in python.

This mirrors the approach used for query profile configuration in
`vespa.configuration.query_profiles` and allows users to build a deployment
descriptor directly in Python, ensuring the top-level tag is `deployment`.

Example:

    from vespa.configuration.deployment import (
        deployment, prod, region
    )
    from vespa.package import ApplicationPackage

    dep_cfg = deployment(
        prod(
            region("aws-us-east-1c"),
            region("aws-us-west-2a")
        ),
        version="1.0"
    )

    app_package = ApplicationPackage(name="myapp", deployment_config=dep_cfg)
    # app_package.deployment_to_text now returns the expected deployment.xml content
    # and will write this to deployment.xml-file on deployment (or if app_package.to_files() is called explicitly)
"""

from typing import List
from vespa.configuration.vt import VT, create_tag_function, voids

# Tags used (subset guided by reference examples). Add more as needed.
deployment_tags: List[str] = [
    "deployment",
    "instance",
    "prod",
    "region",
    "block-change",
    "delay",
    "parallel",
    "steps",
    "endpoints",
    "endpoint",
    "staging",
]

# Fail fast on duplicates
_dup = {t for t in deployment_tags if deployment_tags.count(t) > 1}
if _dup:
    raise ValueError(f"Tags duplicated in deployment_tags: {_dup}")

g = globals()
for _tag in deployment_tags:
    g[VT.sanitize_tag_name(_tag)] = create_tag_function(_tag, _tag in voids)


class DeploymentItem:
    """Wrapper around a VT structure representing a full deployment.xml tree.

    Ensures that the provided VT has a `deployment` top-level tag and exposes
    the rendered XML via the `xml` property.
    """

    def __init__(self, root: VT):
        if not isinstance(root, VT):  # defensive
            raise TypeError(
                f"DeploymentItem expects a VT root element, got {type(root).__name__}"
            )
        if root.tag != VT.sanitize_tag_name("deployment"):
            raise ValueError(
                "Top level tag for deployment configuration must be 'deployment'"
            )
        self.root = root

    @classmethod
    def from_vt(cls, vt_obj: VT) -> "DeploymentItem":
        return cls(vt_obj)

    def to_xml(self) -> str:
        return str(self.root.to_xml()).strip()

    def __repr__(self) -> str:  # pragma: no cover
        return f"DeploymentItem(root={self.root!r})"
