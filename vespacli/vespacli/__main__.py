import sys

from . import run_vespa_cli


def main(*args):
    run_vespa_cli(*args)


if __name__ == "__main__":
    print("Running vespacli.__main__")
    print(f"sys.argv: {sys.argv}")
    main(*sys.argv[1:])
