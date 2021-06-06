# Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

from typing import Callable, List, Optional, Dict


#
# Query property
#
class QueryProperty(object):
    """
    Abstract class for query property.
    """

    def get_query_properties(self, query: Optional[str] = None) -> Dict:
        """
        Extract query property syntax.

        :param query: Query input.
        :return: dict containing the relevant request properties to be included in the query.
        """
        raise NotImplementedError


class QueryRankingFeature(QueryProperty):
    def __init__(
        self,
        name: str,
        mapping: Callable[[str], List[float]],
    ) -> None:
        """
        Include ranking.feature.query into a Vespa query.

        :param name: Name of the feature.
        :param mapping: Function mapping a string to a list of floats.
        """
        super().__init__()
        self.name = name
        self.mapping = mapping

    def get_query_properties(self, query: Optional[str] = None) -> Dict[str, str]:
        value = self.mapping(query)
        return {"ranking.features.query({})".format(self.name): str(value)}


#
# Match phase
#
class MatchFilter(object):
    """
    Abstract class for match filters.
    """

    def create_match_filter(self, query: str) -> str:
        """
        Create part of the YQL expression related to the filter.

        :param query: Query input.
        :return: Part of the YQL expression related to the filter.
        """
        raise NotImplementedError

    def get_query_properties(self, query: Optional[str] = None) -> Dict:
        """
        Relevant request properties associated with the filter.

        :param query: Query input.
        :return: dict containing the relevant request properties associated with the filter.
        """
        raise NotImplementedError


class AND(MatchFilter):
    def __init__(self) -> None:
        """
        Filter that match document containing all the query terms.
        """
        super().__init__()

    def create_match_filter(self, query: str) -> str:
        return '(userInput("{}"))'.format(query)

    def get_query_properties(self, query: Optional[str] = None) -> Dict:
        return {}


class OR(MatchFilter):
    def __init__(self) -> None:
        """
        Filter that match any document containing at least one query term.
        """
        super().__init__()

    def create_match_filter(self, query: str) -> str:
        return '([{{"grammar": "any"}}]userInput("{}"))'.format(query)

    def get_query_properties(self, query: Optional[str] = None) -> Dict:
        return {}


class WeakAnd(MatchFilter):
    def __init__(self, hits: int, field: str = "default") -> None:
        """
        Match documents according to the weakAND algorithm.

        Reference: https://docs.vespa.ai/documentation/using-wand-with-vespa.html

        :param hits: Lower bound on the number of hits to be retrieved.
        :param field: Which Vespa field to search.
        """
        super().__init__()
        self.hits = hits
        self.field = field

    def create_match_filter(self, query: str) -> str:
        query_tokens = query.split(" ")
        terms = ", ".join(
            ['{} contains "{}"'.format(self.field, token) for token in query_tokens]
        )
        return '([{{"targetNumHits": {}}}]weakAnd({}))'.format(self.hits, terms)

    def get_query_properties(self, query: Optional[str] = None) -> Dict:
        return {}


class ANN(MatchFilter):
    def __init__(
        self,
        doc_vector: str,
        query_vector: str,
        hits: int,
        label: str,
        approximate: bool = True,
    ) -> None:
        """
        Match documents according to the nearest neighbor operator.

        Reference: https://docs.vespa.ai/documentation/reference/query-language-reference.html#nearestneighbor

        :param doc_vector: Name of the document field to be used in the distance calculation.
        :param query_vector: Name of the query field to be used in the distance calculation.
        :param hits: Lower bound on the number of hits to return.
        :param label: A label to identify this specific operator instance.
        :param approximate: True to use approximate nearest neighbor and False to use brute force. Default to True.
        """
        super().__init__()
        self.doc_vector = doc_vector
        self.query_vector = query_vector
        self.hits = hits
        self.label = label
        self.approximate = approximate
        self._approximate = "true" if self.approximate is True else "false"

    def create_match_filter(self, query: str) -> str:
        return '([{{"targetNumHits": {}, "label": "{}", "approximate": {}}}]nearestNeighbor({}, {}))'.format(
            self.hits, self.label, self._approximate, self.doc_vector, self.query_vector
        )

    def get_query_properties(self, query: Optional[str] = None) -> Dict[str, str]:
        return {}


class Union(MatchFilter):
    def __init__(self, *args: MatchFilter) -> None:
        """
        Match documents that belongs to the union of many match filters.

        :param args: Match filters to be taken the union of.
        """
        super().__init__()
        self.operators = args

    def create_match_filter(self, query: str) -> str:
        match_filters = []
        for operator in self.operators:
            match_filter = operator.create_match_filter(query=query)
            if match_filter is not None:
                match_filters.append(match_filter)
        return " or ".join(match_filters)

    def get_query_properties(self, query: Optional[str] = None) -> Dict[str, str]:
        query_properties = {}
        for operator in self.operators:
            query_properties.update(operator.get_query_properties(query=query))
        return query_properties


#
# Ranking phase
#
class RankProfile(object):
    def __init__(self, name: str = "default", list_features: bool = False) -> None:
        """
        Define a rank profile.

        :param name: Name of the rank profile as defined in a Vespa search definition.
        :param list_features: Should the ranking features be returned. Either 'true' or 'false'.
        """
        self.name = name
        self.list_features = "false"
        if list_features:
            self.list_features = "true"


class QueryModel(object):
    def __init__(
        self,
        name: str = "default_name",
        query_properties: Optional[List[QueryProperty]] = None,
        match_phase: MatchFilter = AND(),
        rank_profile: RankProfile = RankProfile(),
        body_function: Optional[Callable[[str], Dict]] = None,
    ) -> None:
        """
        Define a query model.

        :param name: Name of the query model. Used to tag model related quantities, like evaluation metrics.
        :param query_properties: Optional list of QueryProperty.
        :param match_phase: Define the match criteria. One of the MatchFilter options available.
        :param rank_profile: Define the rank criteria.
        :param body_function: Function that take query as parameter and returns the body of a Vespa query.
        """
        self.name = name
        self.query_properties = query_properties if query_properties is not None else []
        self.match_phase = match_phase
        self.rank_profile = rank_profile
        self.body_function = body_function

    def create_body(self, query: str) -> Dict[str, str]:
        """
        Create the appropriate request body to be sent to Vespa.

        :param query: Query input.
        :return: dict representing the request body.
        """

        if self.body_function:
            body = self.body_function(query)
            return body

        query_properties = {}
        for query_property in self.query_properties:
            query_properties.update(query_property.get_query_properties(query=query))
        query_properties.update(self.match_phase.get_query_properties(query=query))

        match_filter = self.match_phase.create_match_filter(query=query)

        body = {
            "yql": "select * from sources * where {};".format(match_filter),
            "ranking": {
                "profile": self.rank_profile.name,
                "listFeatures": self.rank_profile.list_features,
            },
        }
        body.update(query_properties)
        return body


