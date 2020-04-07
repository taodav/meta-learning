import argparse

def process_args(args, parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument('--alg', dest='alg', type=str, default='siamese',
                        help='Algorithm to use')

    parser.add_argument('--dataset', dest="dataset", type=str, default="omniglot",
                        help='Dataset name')

    parser.add_argument('--learning-rate', dest="learning_rate",
                        type=float, default=1e-4,
                        help='Learning rate (default: %(default)s)')

    parser.add_argument('--seed', dest='seed', type=int, default=0,
                        help='Seed for reproducibility')
    parameters = parser.parse_args(args)
    return parameters
