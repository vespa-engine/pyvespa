from vespa.configuration.vt import VT, create_tag_function, voids

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

# Generate dynamic tag functions and map them to valid Python names
for tag in queryprofile_tags:
    sanitized_name = VT.sanitize_tag_name(tag)
    _g[sanitized_name] = create_tag_function(tag, tag in voids)
