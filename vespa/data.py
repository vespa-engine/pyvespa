from typing import List, Dict, Optional

import random
import math


def split_labelled_data(
    labelled_data: List[Dict],
    query_prob: float,
    relevant_docs_prob: Optional[float] = None,
    random_seed: Optional[int] = None,
):
    """
    Split labelled data into train and test sets containing partially observed and unobserved queries.

    :param labelled_data: Each Dict contains `query_id`, `query` and `relevant_docs` keys.
    :param query_prob: The probability of moving a query to the test set.
    :param relevant_docs_prob: The probability of moving relevant docs from train queries to partially observed
        test queries.
    :param random_seed: Number to initialize the random number generator.

    :return: Dict containing `train_set` and `unobserved_test_set`. Also contains `partially_observed_test_set` if
        `relevant_docs_prob` is not None.
    """
    if random_seed:
        random.seed(random_seed)

    number_queries = len(labelled_data)

    #
    # Move entire queries to the test set
    #
    test_query_idx = [
        x
        for x in range(number_queries)
        if x
        in random.sample(
            population=range(number_queries), k=math.floor(number_queries * query_prob)
        )
    ]
    test_unobserved = [
        labelled_data[i] for i in range(number_queries) if i in test_query_idx
    ]
    train_set = [
        labelled_data[i] for i in range(number_queries) if i not in test_query_idx
    ]

    result = {}

    if relevant_docs_prob:
        test_partially_observed = []
        for data in train_set:
            number_relevant_docs = len(data["relevant_docs"])
            test_relevant_docs_idx = [
                x
                for x in range(number_relevant_docs)
                if x
                in random.sample(
                    population=range(number_relevant_docs),
                    k=math.floor(number_relevant_docs * relevant_docs_prob),
                )
            ]
            test_data = {k: data[k] for k in data.keys() if k != "relevant_docs"}
            test_data["relevant_docs"] = [
                data["relevant_docs"][i]
                for i in range(number_relevant_docs)
                if i in test_relevant_docs_idx
            ]
            test_partially_observed.append(test_data)
            data["relevant_docs"] = [
                data["relevant_docs"][i]
                for i in range(number_relevant_docs)
                if i not in test_relevant_docs_idx
            ]
        result["partially_observed_test_set"] = test_partially_observed

    result["train_set"] = train_set
    result["unobserved_test_set"] = test_unobserved

    return result
