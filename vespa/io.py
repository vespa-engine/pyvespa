import warnings
from typing import Any, Optional, Dict, List


class VespaResponse(object):
    """
    Class to represent a Vespa HTTP API response.
    """

    def __init__(
        self, json: Dict, status_code: int, url: str, operation_type: str
    ) -> None:
        self.json = json
        self.status_code = status_code
        self.url = url
        self.operation_type = operation_type

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (
            self.json == other.json
            and self.status_code == other.status_code
            and self.url == other.url
            and self.operation_type == other.operation_type
        )

    def get_status_code(self) -> int:
        """Return status code of the response."""
        return self.status_code

    def is_successfull(self) -> bool:
        """[Deprecated] Use is_successful() instead"""
        warnings.warn(
            "is_successfull is deprecated, use is_successful() instead.",
            DeprecationWarning,
        )
        return self.status_code == 200

    def is_successful(self) -> bool:
        """True if status code is 200."""
        return self.status_code == 200

    def get_json(self) -> Dict:
        """Return json of the response."""
        return self.json


class VespaQueryResponse(VespaResponse):
    def __init__(self, json, status_code, url, request_body=None) -> None:
        super().__init__(
            json=json, status_code=status_code, url=url, operation_type="query"
        )
        self._request_body = request_body

    @property
    def request_body(self) -> Optional[Dict]:
        return self._request_body

    @property
    def hits(self) -> List:
        return self.json.get("root", {}).get("children", [])

    @property
    def number_documents_retrieved(self) -> int:
        return self.json.get("root", {}).get("fields", {}).get("totalCount", 0)

    @property
    def number_documents_indexed(self) -> int:
        return self.json.get("root", {}).get("coverage", {}).get("documents", 0)

    def get_json(self) -> Dict:
        """
        For debugging when the response does not have hits.

        :return: JSON object with full response
        """
        return self.json


class VespaVisitResponse(VespaResponse):
    def __init__(self, json, status_code, url) -> None:
        super().__init__(
            json=json, status_code=status_code, url=url, operation_type="visit"
        )

    @property
    def continuation(self) -> Optional[str]:
        return self.json.get("continuation")

    @property
    def path_id(self) -> str:
        return self.json.get("pathId", "")

    @property
    def documents(self) -> List[Dict[str, Any]]:
        return self.json.get("documents", [])

    @property
    def number_documents_retrieved(self) -> int:
        return self.json.get("documentCount", 0)
