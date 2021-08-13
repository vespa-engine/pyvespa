from typing import Mapping
from collections import OrderedDict


from vespa.package import ApplicationPackage


class ModelServer(ApplicationPackage):
    def __init__(self, name: str):
        """
        Create a Vespa stateless model evaluation server.

        A Vespa stateless model evaluation server is a simplified Vespa application without content clusters.

        :param name: Application name.
        """
        super().__init__(
            name=name,
            schema=None,
            query_profile=None,
            query_profile_type=None,
            stateless_model_evaluation=True,
            create_schema_by_default=False,
            create_query_profile_by_default=False,
        )

    @staticmethod
    def from_dict(mapping: Mapping) -> "ModelServer":
        return ModelServer(name=mapping["name"])

    @property
    def to_dict(self) -> Mapping:
        map = {"name": self.name}
        return map
