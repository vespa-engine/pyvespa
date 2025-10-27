from vespa.application import Vespa
from vespa.evaluation import NearestNeighborHitratioComputer
from vespa.evaluation import NearestNeighborParameterOptimizer

import argparse
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

class VespaANNParameters:
    FORCE_EXACT = {
        "ranking.matching.approximateThreshold": 1.00
    }
    FORCE_HNSW = {
        "ranking.matching.approximateThreshold": 0.00,
        "ranking.matching.filterFirstThreshold": 0.00,
        "ranking.matching.filterFirstExploration": 0.30,
        "ranking.matching.explorationSlack": 0.00
    }
    FORCE_FILTER_FIRST = {
        "ranking.matching.approximateThreshold": 0.00,
        "ranking.matching.filterFirstThreshold": 1.00,
        "ranking.matching.filterFirstExploration": 0.30,
        "ranking.matching.explorationSlack": 0.00
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Vespa ANN Parameter Tool',
        description='Suggests Vespa ANN parameters based on measurements performed on a set of queries.')
    parser.add_argument('url', help="URL.")
    parser.add_argument('port', help="Port.")
    parser.add_argument('query_file', help="File containing queries in GET format (one per line).")
    parser.add_argument('hits', help="Number of target hits used in queries.")
    parser.add_argument('--debug', action='store_true', help="Print debug messages and plot debug graphs.")
    parser.add_argument('--plot', action='store_true', help="Plot results using matplotlib.")
    parser.add_argument('--cert')
    parser.set_defaults(cert=None)
    parser.add_argument('--key')
    parser.set_defaults(key=None)
    args = parser.parse_args()

    app = Vespa(url=args.url, port=args.port, cert=args.cert, key=args.key)

    # Read query file with get queries
    with open(args.query_file) as file:
        get_queries = file.read().splitlines()

    # Parse get queries
    queries = list(map(query_from_get_string, get_queries))

    print("1. Determining hit ratios of queries")
    hitratio_computer = NearestNeighborHitratioComputer(queries, app)
    hitratios = hitratio_computer.run()
    hitratios = list(map(lambda list: list[0], hitratios))

    # Sort hit ratios into buckets
    optimizer = NearestNeighborParameterOptimizer(app, int(args.hits), print_progress=True)
    optimizer.distribute_to_buckets(zip(queries, hitratios))

    # Import matplotlib
    if args.plot:
        import matplotlib.pyplot as plt

    if args.plot:
        x, y = optimizer.get_query_distribution()
        plt.bar(x, y, width=optimizer.get_bucket_interval_width(), align='edge')
        plt.title("Number of queries per fraction filtered out")
        plt.xlabel('Fraction filtered out')
        plt.ylabel('Number of queries')
        axs = plt.gca()
        axs.set_xlim(xmin=0, xmax=1)
        axs.set_ylim(ymin=0)
        plt.show()

    print("2. Initial benchmarks: Exact, HNSW, HNSW (Filter First)")
    benchmark_exact = optimizer.benchmark(**VespaANNParameters.FORCE_EXACT)
    benchmark_hnsw = optimizer.benchmark(**VespaANNParameters.FORCE_HNSW)
    benchmark_filter_first = optimizer.benchmark(**VespaANNParameters.FORCE_FILTER_FIRST)

    if args.plot and args.debug:
        plt.plot(optimizer.buckets_to_filtered_out(benchmark_exact.x), benchmark_exact.y, label="Exact")
        plt.plot(optimizer.buckets_to_filtered_out(benchmark_hnsw.x), benchmark_hnsw.y, label="HNSW")
        plt.plot(optimizer.buckets_to_filtered_out(benchmark_filter_first.x), benchmark_filter_first.y, label="HNSW (Filter First)")

        plt.title("Response time: Debug")
        plt.xlabel('Fraction filtered out')
        plt.ylabel('Response time')
        plt.legend()
        axs = plt.gca()
        axs.set_xlim(xmin=0, xmax=1)
        axs.set_ylim(ymin=0)
        plt.show()

    print("3. Initial recall measurements: HNSW, HNSW (Filter First)")
    recall_hnsw = optimizer.compute_average_recalls(**VespaANNParameters.FORCE_HNSW)
    recall_filter_first = optimizer.compute_average_recalls(**VespaANNParameters.FORCE_FILTER_FIRST)

    if args.plot and args.debug:
        plt.plot(optimizer.buckets_to_filtered_out(recall_hnsw.x), recall_hnsw.y, label="HNSW")
        plt.plot(optimizer.buckets_to_filtered_out(recall_filter_first.x), recall_filter_first.y, label="HNSW (Filter First)")

        plt.title("Recall: Debug")
        plt.xlabel('Fraction filtered out')
        plt.ylabel('Recall')
        plt.legend()
        axs = plt.gca()
        axs.set_xlim(xmin=0, xmax=1)
        axs.set_ylim(ymin=0, ymax=1)
        plt.show()

    print("4. Determining suggestion for filterFirstThreshold")
    filter_first_threshold = optimizer.determine_filter_first_threshold(benchmark_hnsw, benchmark_filter_first)
    print(f"  filterFirstThreshold: {round(optimizer.bucket_to_hitratio(filter_first_threshold), 3)}")

    print("5. Determining suggestion for approximateThreshold")
    approximate_threshold = optimizer.determine_approximate_threshold(benchmark_exact, benchmark_hnsw, benchmark_filter_first, filter_first_threshold)
    print(f"  approximateThreshold: {round(optimizer.bucket_to_hitratio(approximate_threshold), 3)}")

    print("6. Determining suggestion for filterFirstExploration")
    # Filter first exploration is not specified in buckets
    filter_first_exploration = optimizer.determine_filter_first_exploration(filter_first_threshold, approximate_threshold)
    print(f"  filterFirstExploration: {round(filter_first_exploration, 3)}")

    parameters_optimized = {
        "ranking.matching.approximateThreshold": optimizer.bucket_to_hitratio(approximate_threshold),
        "ranking.matching.filterFirstThreshold": optimizer.bucket_to_hitratio(filter_first_threshold),
        "ranking.matching.filterFirstExploration": filter_first_exploration
    }

    print("7. Comparing current to optimized settings")
    benchmark_current = optimizer.benchmark()
    benchmark_optimized = optimizer.benchmark(**parameters_optimized)

    if args.plot:
        plt.plot(optimizer.buckets_to_filtered_out(benchmark_current.x), benchmark_current.y, label="Current")
        plt.plot(optimizer.buckets_to_filtered_out(benchmark_optimized.x), benchmark_optimized.y, label="Optimized")

        plt.title("Response time: Current vs. Optimized")
        plt.xlabel('Fraction filtered out')
        plt.ylabel('Response time')
        plt.legend()
        axs = plt.gca()
        axs.set_xlim(xmin=0, xmax=1)
        axs.set_ylim(ymin=0)
        plt.show()

    recall_current = optimizer.compute_average_recalls()
    recall_optimized = optimizer.compute_average_recalls(**parameters_optimized)

    if args.plot:
        plt.plot(optimizer.buckets_to_filtered_out(recall_current.x), recall_current.y, label="Current")
        plt.plot(optimizer.buckets_to_filtered_out(recall_optimized.x), recall_optimized.y, label="Optimized")

        plt.title("Recall: Current vs. Optimized")
        plt.xlabel('Fraction filtered out')
        plt.ylabel('Recall')
        plt.legend()
        axs = plt.gca()
        axs.set_xlim(xmin=0, xmax=1)
        axs.set_ylim(ymin=0, ymax=1)
        plt.show()
