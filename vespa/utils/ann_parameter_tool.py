# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

from vespa.application import Vespa
from vespa.evaluation import VespaNNGlobalFilterHitratioEvaluator
from vespa.evaluation import VespaNNParameterOptimizer

import argparse
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
    parser.add_argument(
        "--filterFirstExploration",
        help="Do not suggest filterFirstExploration, but use this value instead.",
    )
    parser.set_defaults(filterFirstExploration=None)
    parser.add_argument(
        "--filterFirstThreshold",
        help="Do not suggest filterFirstThreshold, but use this value instead.",
    )
    parser.set_defaults(filterFirstThreshold=None)
    parser.add_argument(
        "--approximateThreshold",
        help="Do not suggest approximateThreshold, but use this value instead.",
    )
    parser.set_defaults(approximateThreshold=None)
    parser.add_argument(
        "--postFilterThreshold",
        help="Do not suggest postFilterThreshold, but use this value instead.",
    )
    parser.set_defaults(postFilterThreshold=None)
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
    hitratio_evaluator = VespaNNGlobalFilterHitratioEvaluator(queries, app, verify_target_hits=int(args.hits))
    hitratio_list = hitratio_evaluator.run()

    for i in range(0, len(hitratio_list)):
        hitratios = hitratio_list[i]
        if len(hitratios) == 0:
            sys.exit(f"Aborting: No hit ratio found for query #{i} (No nearestNeighbor operator?)")
        if len(hitratios) > 1:
            sys.exit(f"Aborting: More than one hit ratio found for query #{i} (Multiple nearestNeighbor operators?)")

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
    # filterFirstExploration
    ####################################################################################################################
    if args.filterFirstExploration is None:
        print("Determining suggestion for filterFirstExploration")
        # Filter first exploration is not specified in buckets
        filter_first_exploration, benchmarks = (
            optimizer.suggest_filter_first_exploration()
        )
        print(f"  filterFirstExploration: {round(filter_first_exploration, 3)}")

        if args.plot and args.debug:
            for ffe, benchmark, _ in benchmarks:
                plt.plot(
                    optimizer.buckets_to_filtered_out(benchmark.x),
                    benchmark.y,
                    label=f"filterFirstExploration = {ffe}",
                )

            plt.title("Response time: filterFirstExploration")
            plt.xlabel("Fraction filtered out")
            plt.ylabel("Response time")
            plt.legend()
            axs = plt.gca()
            axs.set_xlim(xmin=0, xmax=1)
            axs.set_ylim(ymin=0)
            plt.show()

            for ffe, _, recall in benchmarks:
                plt.plot(
                    optimizer.buckets_to_filtered_out(recall.x),
                    recall.y,
                    label=f"filterFirstExploration = {ffe}",
                )

            plt.title("Recall: filterFirstExploration")
            plt.xlabel("Fraction filtered out")
            plt.ylabel("Recall")
            plt.legend()
            axs = plt.gca()
            axs.set_xlim(xmin=0, xmax=1)
            axs.set_ylim(ymin=0, ymax=1)
            plt.show()
    else:
        filter_first_exploration = float(args.filterFirstExploration)
        print(
            f"Using supplied value for filterFirstExploration: {filter_first_exploration}"
        )

    ####################################################################################################################
    # filterFirstThreshold
    ####################################################################################################################
    if args.filterFirstThreshold is None:
        print("Determining suggestion for filterFirstThreshold")
        hnsw_parameters = {
            "timeout": "20s",
            "ranking.matching.approximateThreshold": 0.00,
            "ranking.matching.filterFirstThreshold": 0.00,
        }
        benchmark_hnsw = optimizer.benchmark(**hnsw_parameters)

        filter_first_parameters = {
            "timeout": "20s",
            "ranking.matching.approximateThreshold": 0.00,
            "ranking.matching.filterFirstThreshold": 1.00,
            "ranking.matching.filterFirstExploration": filter_first_exploration,
        }
        benchmark_filter_first = optimizer.benchmark(**filter_first_parameters)

        if args.plot and args.debug:
            plt.plot(
                optimizer.buckets_to_filtered_out(benchmark_hnsw.x),
                benchmark_hnsw.y,
                label="HNSW",
            )
            plt.plot(
                optimizer.buckets_to_filtered_out(benchmark_filter_first.x),
                benchmark_filter_first.y,
                label="HNSW (Filter First)",
            )

            plt.title("Response time: filterFirstThreshold")
            plt.xlabel("Fraction filtered out")
            plt.ylabel("Response time")
            plt.legend()
            axs = plt.gca()
            axs.set_xlim(xmin=0, xmax=1)
            axs.set_ylim(ymin=0)
            plt.show()

        filter_first_threshold = optimizer.suggest_filter_first_threshold(
            benchmark_hnsw, benchmark_filter_first
        )
        print(f"  filterFirstThreshold: {round(filter_first_threshold, 3)}")
    else:
        filter_first_threshold = float(args.filterFirstThreshold)
        print(
            f"Using supplied value for filterFirstThreshold: {filter_first_threshold}"
        )

    ####################################################################################################################
    # approximateThreshold
    ####################################################################################################################
    if args.approximateThreshold is None:
        print("Determining suggestion for approximateThreshold")
        exact_parameters = {
            "timeout": "20s",
            "ranking.matching.approximateThreshold": 1.00,
        }
        benchmark_exact = optimizer.benchmark(**exact_parameters)

        filter_first_parameters2 = {
            "timeout": "20s",
            "ranking.matching.approximateThreshold": 0.00,
            "ranking.matching.filterFirstThreshold": filter_first_threshold,
            "ranking.matching.filterFirstExploration": filter_first_exploration,
        }
        benchmark_filter_first2 = optimizer.benchmark(**filter_first_parameters2)

        if args.plot and args.debug:
            plt.plot(
                optimizer.buckets_to_filtered_out(benchmark_exact.x),
                benchmark_exact.y,
                label="Exact",
            )
            plt.plot(
                optimizer.buckets_to_filtered_out(benchmark_filter_first2.x),
                benchmark_filter_first2.y,
                label="HNSW (Filter First) optimized",
            )

            plt.title("Response time: approximateThreshold")
            plt.xlabel("Fraction filtered out")
            plt.ylabel("Response time")
            plt.legend()
            axs = plt.gca()
            axs.set_xlim(xmin=0, xmax=1)
            axs.set_ylim(ymin=0)
            plt.show()

        approximate_threshold = optimizer.suggest_approximate_threshold(
            benchmark_exact, benchmark_filter_first2
        )
        print(f"  approximateThreshold: {round(approximate_threshold, 3)}")
    else:
        approximate_threshold = float(args.approximateThreshold)
        print(f"Using supplied value for approximateThreshold: {approximate_threshold}")

    ####################################################################################################################
    # postFilterThreshold
    ####################################################################################################################
    if args.postFilterThreshold is None:
        print("Determining suggestion for postFilterThreshold")
        post_filtering_parameters = {
            "timeout": "20s",
            "ranking.matching.postFilterThreshold": 0.00,
            "ranking.matching.approximateThreshold": 0.00,
        }
        benchmark_post_filtering = optimizer.benchmark(**post_filtering_parameters)

        filter_first_parameters3 = {
            "timeout": "20s",
            "ranking.matching.approximateThreshold": approximate_threshold,
            "ranking.matching.filterFirstThreshold": filter_first_threshold,
            "ranking.matching.filterFirstExploration": filter_first_exploration,
        }
        benchmark_filter_first3 = optimizer.benchmark(**filter_first_parameters3)

        if args.plot and args.debug:
            plt.plot(
                optimizer.buckets_to_filtered_out(benchmark_post_filtering.x),
                benchmark_post_filtering.y,
                label="Post-Filtering",
            )
            plt.plot(
                optimizer.buckets_to_filtered_out(benchmark_filter_first3.x),
                benchmark_filter_first3.y,
                label="HNSW (Filter First) optimized",
            )

            plt.title("Response time: postFilterThreshold")
            plt.xlabel("Fraction filtered out")
            plt.ylabel("Response time")
            plt.legend()
            axs = plt.gca()
            axs.set_xlim(xmin=0, xmax=1)
            axs.set_ylim(ymin=0)
            plt.show()

        recall_post_filtering = optimizer.compute_average_recalls(
            **post_filtering_parameters
        )
        recall_filter_first3 = optimizer.compute_average_recalls(
            **filter_first_parameters3
        )

        if args.plot and args.debug:
            plt.plot(
                optimizer.buckets_to_filtered_out(recall_post_filtering.x),
                recall_post_filtering.y,
                label="Post-Filtering",
            )
            plt.plot(
                optimizer.buckets_to_filtered_out(recall_filter_first3.x),
                recall_filter_first3.y,
                label="HNSW (Filter First) optimized",
            )

            plt.title("Recall: postFilterThreshold")
            plt.xlabel("Fraction filtered out")
            plt.ylabel("Recall")
            plt.legend()
            axs = plt.gca()
            axs.set_xlim(xmin=0, xmax=1)
            axs.set_ylim(ymin=0, ymax=1)
            plt.show()

        post_filter_threshold = optimizer.suggest_post_filter_threshold(
            benchmark_post_filtering,
            recall_post_filtering,
            benchmark_filter_first3,
            recall_filter_first3,
        )
        print(f"  postFilterThreshold: {round(post_filter_threshold, 3)}")
    else:
        post_filter_threshold = float(args.postFilterThreshold)
        print(f"Using supplied value for postFilterThreshold: {post_filter_threshold}")

    ####################################################################################################################
    # Comparison
    ####################################################################################################################
    print("Comparing current to suggested settings")

    suggested_parameters = {
        "timeout": "20s",
        "ranking.matching.approximateThreshold": approximate_threshold,
        "ranking.matching.filterFirstThreshold": filter_first_threshold,
        "ranking.matching.filterFirstExploration": filter_first_exploration,
        "ranking.matching.postFilterThreshold": post_filter_threshold,
    }

    benchmark_current = optimizer.benchmark()
    benchmark_suggested = optimizer.benchmark(**suggested_parameters)

    if args.plot:
        plt.plot(
            optimizer.buckets_to_filtered_out(benchmark_current.x),
            benchmark_current.y,
            label="Current",
        )
        plt.plot(
            optimizer.buckets_to_filtered_out(benchmark_suggested.x),
            benchmark_suggested.y,
            label="Suggested",
        )

        plt.title("Response time: Current vs. Suggested")
        plt.xlabel("Fraction filtered out")
        plt.ylabel("Response time")
        plt.legend()
        axs = plt.gca()
        axs.set_xlim(xmin=0, xmax=1)
        axs.set_ylim(ymin=0)
        plt.show()

    recall_current = optimizer.compute_average_recalls()
    recall_optimized = optimizer.compute_average_recalls(**suggested_parameters)

    if args.plot:
        plt.plot(
            optimizer.buckets_to_filtered_out(recall_current.x),
            recall_current.y,
            label="Current",
        )
        plt.plot(
            optimizer.buckets_to_filtered_out(recall_optimized.x),
            recall_optimized.y,
            label="Optimized",
        )

        plt.title("Recall: Current vs. Optimized")
        plt.xlabel("Fraction filtered out")
        plt.ylabel("Recall")
        plt.legend()
        axs = plt.gca()
        axs.set_xlim(xmin=0, xmax=1)
        axs.set_ylim(ymin=0, ymax=1)
        plt.show()
