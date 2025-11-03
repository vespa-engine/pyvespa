# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

from vespa.application import Vespa

import argparse
import urllib.parse

from vespa.evaluation import VespaNNRecallEvaluator


def query_from_get_string(get_query):
    url = urllib.parse.urlparse(get_query)
    assert url.path == "/search/"
    parsed_query = urllib.parse.parse_qs(url.query)
    query = {}
    for key in parsed_query.keys():
        query[key] = parsed_query[key][0]

    assert "yql" in query
    return query


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Vespa ANN Recall Tool", description="Determines recall of ANN queries."
    )
    parser.add_argument("url", help="URL.")
    parser.add_argument("port", help="Port.")
    parser.add_argument(
        "query_file", help="File containing queries in GET format (one per line)."
    )
    parser.add_argument("hits", help="Number of target hits used in queries.")
    parser.add_argument("--silent", action="store_true")
    parser.set_defaults(silent=False)
    parser.add_argument("--cert")
    parser.set_defaults(cert=None)
    parser.add_argument("--key")
    parser.set_defaults(key=None)
    args = parser.parse_args()

    app = Vespa(url=args.url, port=args.port, cert=args.cert, key=args.key)

    # Read query file with get queries
    with open(args.query_file) as file:
        get_queries = file.read().splitlines()

    # Parse get queries
    queries = list(map(query_from_get_string, get_queries))

    # Compute hit ratios
    if not args.silent:
        print("Determining recalls")
    recall_evaluator = VespaNNRecallEvaluator(queries, int(args.hits), app)
    recalls = recall_evaluator.run()

    # Print hit ratios
    if args.silent:
        for recall in recalls:
            print(recall)
    else:
        print("Recalls:")
        for recall in recalls:
            print(f"  {recall}")
