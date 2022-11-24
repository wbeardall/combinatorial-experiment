import argparse

from .combinatorial_experiment import CombinatorialExperiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    CombinatorialExperiment.resume(args.path)
