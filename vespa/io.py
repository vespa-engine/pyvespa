from typing import Optional, Dict, List

from pandas import DataFrame


class VespaResponse(object):
    def __init__(self, json, status_code, url, operation_type):
        self.json = json
        self.status_code = status_code
        self.url = url
        self.operation_type = operation_type

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return (
            self.json == other.json
            and self.status_code == other.status_code
            and self.url == other.url
            and self.operation_type == other.operation_type
        )


def trec_format(
    vespa_result, id_field: Optional[str] = None, qid: int = 0
) -> DataFrame:
    """
    Function to format Vespa output according to TREC format.

    TREC format include qid, doc_id, score and rank.

    :param vespa_result: raw Vespa result from query.
    :param id_field: Name of the Vespa field to use as 'doc_id' value.
    :param qid: custom query id.
    :return: pandas DataFrame with columns qid, doc_id, score and rank.
    """
    hits = vespa_result.get("root", {}).get("children", [])
    records = []
    for rank, hit in enumerate(hits):
        records.append(
            {
                "qid": qid,
                "doc_id": hit["fields"][id_field]
                if id_field is not None
                else hit["id"],
                "score": hit["relevance"],
                "rank": rank,
            }
        )
    return DataFrame.from_records(records)


class VespaQueryResponse(VespaResponse):
    def __init__(self, json, status_code, url, request_body=None):
        super().__init__(json=json, status_code=status_code, url=url, operation_type="query")
        self._request_body = request_body

    @property
    def request_body(self) -> Optional[Dict]:
        return self._request_body

    @property
    def hits(self) -> List:
        return self.json.get("root", {}).get("children", [])

    def get_hits(self, format_function=trec_format, **kwargs):
        """
        Get Vespa hits according to `format_function` format.

        :param format_function: function to format the raw Vespa result. Should take raw vespa result as first argument.
        :param kwargs: Extra arguments to be passed to `format_function`.
        :return: Output of the `format_function`.
        """
        if not format_function:
            return self.hits
        return format_function(self.json, **kwargs)

    @property
    def number_documents_retrieved(self) -> int:
        return self.json.get("root", {}).get("fields", {}).get("totalCount", 0)

    @property
    def number_documents_indexed(self) -> int:
        return (
            self.json.get("root", {}).get("coverage", {}).get("documents", 0)
        )
