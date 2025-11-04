# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

from vespa.application import Vespa
from vespa.evaluation import VespaNNGlobalFilterHitratioEvaluator
from vespa.evaluation import VespaNNParameterOptimizer

import argparse
import json
import sys
import urllib.parse


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
        prog="Vespa ANN Parameter Tool",
        description="Suggests Vespa ANN parameters based on measurements performed on a set of queries.",
    )
    parser.add_argument("url", help="URL.")
    parser.add_argument("port", help="Port.")
    parser.add_argument(
        "query_file", help="File containing queries in GET format (one per line)."
    )
    parser.add_argument("hits", help="Number of target hits used in queries.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug messages and plot debug graphs.",
    )
    parser.add_argument(
        "--plot", action="store_true", help="Plot results using matplotlib."
    )
    parser.add_argument("--cert")
    parser.set_defaults(cert=None)
    parser.add_argument("--key")
    parser.set_defaults(key=None)
    args = parser.parse_args()

    # Import matplotlib
    if args.plot:
        import matplotlib.pyplot as plt

    app = Vespa(url=args.url, port=args.port, cert=args.cert, key=args.key)

    # Read query file with get queries
    with open(args.query_file) as file:
        get_queries = file.read().splitlines()

    # Parse get queries
    queries = list(map(query_from_get_string, get_queries))

    ####################################################################################################################
    # Hit ratios
    ####################################################################################################################
    print("Determining hit ratios of queries")
    hitratio_evaluator = VespaNNGlobalFilterHitratioEvaluator(
        queries, app, verify_target_hits=int(args.hits)
    )
    hitratio_list = hitratio_evaluator.run()

    for i in range(0, len(hitratio_list)):
        hitratios = hitratio_list[i]
        if len(hitratios) == 0:
            sys.exit(
                f"Aborting: No hit ratio found for query #{i} (No nearestNeighbor operator?)"
            )
        if len(hitratios) > 1:
            sys.exit(
                f"Aborting: More than one hit ratio found for query #{i} (Multiple nearestNeighbor operators?)"
            )

    hitratios = list(map(lambda list: list[0], hitratio_list))

    # Sort hit ratios into buckets
    optimizer = VespaNNParameterOptimizer(app, int(args.hits), print_progress=True)
    optimizer.distribute_to_buckets(zip(queries, hitratios))

    if args.plot:
        x, y = optimizer.get_query_distribution()
        plt.bar(x, y, width=optimizer.get_bucket_interval_width(), align="edge")
        plt.title("Number of queries per fraction filtered out")
        plt.xlabel("Fraction filtered out")
        plt.ylabel("Number of queries")
        axs = plt.gca()
        axs.set_xlim(xmin=0, xmax=1)
        axs.set_ylim(ymin=0)
        plt.show()

    if not optimizer.has_sufficient_queries():
        print(
            "  Warning: Selection of queries does not cover enough hit ratios to get meaningful results."
        )

    if not optimizer.buckets_sufficiently_filled():
        print("  Warning: Only few queries for a specific hit ratio.")

    ####################################################################################################################
    # Parameter Optimization
    ####################################################################################################################
    report = optimizer.run()
    suggested_parameters = {
        "ranking.matching.approximateThreshold": report["approximateThreshold"][
            "suggestion"
        ],
        "ranking.matching.filterFirstThreshold": report["filterFirstThreshold"][
            "suggestion"
        ],
        "ranking.matching.filterFirstExploration": report["filterFirstExploration"][
            "suggestion"
        ],
        "ranking.matching.postFilterThreshold": report["postFilterThreshold"][
            "suggestion"
        ],
    }
    print("Suggested parameters:")
    print(json.dumps(suggested_parameters, sort_keys=True, indent=4))

    if args.debug:
        print("Full report:")
        print(json.dumps(report, sort_keys=True, indent=4))

    def plot_benchmarks(title, benchmarks):
        for key, benchmark in benchmarks.items():
            plt.plot(
                optimizer.get_filtered_out_ratios(),
                benchmark,
                label=f"{key}",
            )

        plt.title(f"Response time: {title}")
        plt.xlabel("Fraction filtered out")
        plt.ylabel("Response time")
        plt.legend()
        axs = plt.gca()
        axs.set_xlim(xmin=0, xmax=1)
        axs.set_ylim(ymin=0)
        plt.show()

    def plot_recall_measurements(title, recall_measurements):
        for key, recall_measurement in recall_measurements.items():
            plt.plot(
                optimizer.get_filtered_out_ratios(),
                recall_measurement,
                label=f"{key}",
            )

        plt.title(f"Recall: {title}")
        plt.xlabel("Fraction filtered out")
        plt.ylabel("Recall")
        plt.legend()
        axs = plt.gca()
        axs.set_xlim(xmin=0, xmax=1)
        axs.set_ylim(ymin=0, ymax=1)
        plt.show()

    # Plot debug information that led to suggestions
    if args.plot and args.debug:
        for parameter in [
            "filterFirstExploration",
            "filterFirstThreshold",
            "approximateThreshold",
            "postFilterThreshold",
        ]:
            parameter_report = report[parameter]
            if "benchmarks" in parameter_report:
                plot_benchmarks(parameter, parameter_report["benchmarks"])
            if "recall_measurements" in parameter_report:
                plot_recall_measurements(
                    parameter, parameter_report["recall_measurements"]
                )

    ####################################################################################################################
    # Comparison
    ####################################################################################################################
    print("Comparing current to suggested settings")

    timeout = {
        "timeout": "20s",
    }

    benchmarks = {
        "Current": optimizer.benchmark(**timeout).y,
        "Suggestions": optimizer.benchmark(**dict(suggested_parameters, **timeout)).y,
    }
    recall_measurements = {
        "Current": optimizer.compute_average_recalls(**timeout).y,
        "Suggestions": optimizer.compute_average_recalls(
            **dict(suggested_parameters, **timeout)
        ).y,
    }

    print("Benchmarks:")
    print(json.dumps(benchmarks, sort_keys=True, indent=4))

    print("Recall Measurements:")
    print(json.dumps(benchmarks, sort_keys=True, indent=4))

    if args.plot:
        plot_benchmarks("Current vs Suggestions", benchmarks)
        plot_recall_measurements("Current vs Suggestions", recall_measurements)
