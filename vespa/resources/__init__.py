import os


def get_resource_path(resource_filename: str) -> str:
    """
    Get the path to a resource file.

    :param resource_name: Name of the resource file.
    :return: Path to the resource file.
    """
    return os.path.join(os.path.dirname(__file__), resource_filename)
