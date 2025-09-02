from vespa.configuration import services, query_profiles, deployment
from vespa.configuration.services import generated_services_tags
from vespa.configuration.query_profiles import generated_query_profile_tags
from vespa.configuration.deployment import generated_deployment_tags

for mod, tag_set in [
    (services, generated_services_tags),
    (query_profiles, generated_query_profile_tags),
    (deployment, generated_deployment_tags),
]:
    mod_location = mod.__file__
    file_location = mod.__file__
    pyi_file = file_location.replace(".py", ".pyi")
    with open(pyi_file, "w") as f:
        # write import statement for VT
        f.write("from vespa.configuration.vt import VT\n\n")
        for tag in tag_set:
            f.write(f"def {tag}(*c, **kwargs) -> VT: ...\n")
