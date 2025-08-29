from vespa.configuration.vt import VT, create_tag_function, voids
from typing import NamedTuple

# NB! If you modify this, please run 'python vespa/configuration/generate_pyi.py' to regenerate the .pyi stub files.
queryprofile_tags = [
    "query-profile",
    "query-profile-type",
    "field",
    "match",
    "strict",
    "description",
    "dimensions",
    "ref",
]
# Fail if any tag is duplicated. Provide feedback of which tags are duplicated.
duplicated = set([tag for tag in queryprofile_tags if queryprofile_tags.count(tag) > 1])
if duplicated:
    raise ValueError(f"Tags duplicated in queryprofile_tags: {duplicated}")
# Mapping general XML tags
_g = globals()

generated_query_profile_tags = set()
# Generate dynamic tag functions and map them to valid Python names
for tag in queryprofile_tags:
    sanitized_name = VT.sanitize_tag_name(tag)
    generated_query_profile_tags.add(sanitized_name)
    _g[sanitized_name] = create_tag_function(tag, tag in voids)


class QueryProfileItem(NamedTuple):
    tag: str
    id_: str
    xml: str

    @classmethod
    def from_vt(cls, vt_config: VT) -> "QueryProfileItem":
        """Create a QueryProfileItem from a VT configuration object."""
        if not isinstance(vt_config, VT):
            raise TypeError(
                f"vt_config must be an instance of VT, got {type(vt_config).__name__}"
            )
        tag = vt_config.tag
        id_ = vt_config.get("id")

        # Validate
        if tag not in ["query_profile", "query_profile_type"]:
            raise ValueError(
                f"Query profile item must be of type 'query_profile' or 'query_profile_type', got '{tag}'"
            )

        if not id_ or not str(id_).strip():
            raise ValueError(
                f"Query profile item of type '{tag}' must have a non-empty 'id'"
            )

        clean_id = str(id_).strip()
        xml = vt_config.to_xml()

        return cls(tag, clean_id, xml)

    def to_xml(self) -> str:
        """Convert the QueryProfileItem to its XML representation."""
        return str(self.xml).strip()

    def __str__(self) -> str:
        return self.xml
