from typing import Mapping


from vespa.package import ApplicationPackage


class ModelServer(ApplicationPackage):
    def __init__(self, name: str, model_file_path: str):
        """
        Create a Vespa stateless model evaluation server.

        A Vespa stateless model evaluation server is a simplified Vespa application without content clusters.

        :param name: Application name.
        :param model_file_path: The path to the .onnx file to include in the application package.
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
        self.model_file_path = model_file_path

    @staticmethod
    def from_dict(mapping: Mapping) -> "ModelServer":
        return ModelServer(name=mapping["name"])

    @property
    def to_dict(self) -> Mapping:
        map = {"name": self.name}
        return map
