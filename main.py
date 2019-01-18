from cas2vec_helper import (build_config, load_cascades,
                            process_observation_prediction_events)
from cas2vec import run_cas2vec, cas2vec_model
import argparse


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cas-path', type=str, default='',
                        help='A path to the cascade file. Default is empty')
    parser.add_argument('--model-dir', type=str, default='./models',
                        help='A path to a model directory. If specified, the best model '
                             'based on its performance on the development set will be '
                             'saved to this directory. Default is empty')
    parser.add_argument('--time-unit', type=str, default='h',
                        help='A time unit associated to all the time arguments, '
                             'h - hours, m - minutes, s - seconds.'
                             'Default is h.')
    parser.add_argument('--disc-method', type=str, default='counter',
                        help='Discretization method, counter or const(disc in the '
                             'original paper). Default is counter')
    parser.add_argument('--obs-time', type=int, default=1,
                        help='An observation time, time according to --time-unit. '
                             'Default is 1 hour')
    parser.add_argument('--prd-time', type=int, default=16,
                        help='A prediction time = obs_time + delta, delta > 0. '
                             'Default is 16 hours')
    parser.add_argument('--num-bins', type=int, default=40,
                        help='The number of bins(slices)')
    parser.add_argument('--seq-len', type=int, nargs=40,
                        help='The length of the sequence that is used as input '
                             'to the CNN. If --disc-method is counter, this is '
                             'the same as --num-bins')
    parser.add_argument('--threshold', type=int, default=1000,
                        help='A virality threshold. Default is 1000')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size. Default is 32')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs. Default is 10')
    parser.add_argument('--size', type=int, default=128,
                        help='Embedding size for the input cascade matrix. '
                             'Default is 128')
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='Dropout probability at the output layer. '
                             'Default is 0.4')
    parser.add_argument('--filters', nargs='*', type=int,
                        help='A list of filter sizes')
    parser.add_argument('--kernel-size', nargs='*', type=int,
                        help='A list of kernel sizes')
    parser.add_argument('--fcc-layers', nargs='*', type=int,
                        help='A list of values for the fully connected layers'
                             'after the convolution and max pooling')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate. Default is 0.0001')
    parser.add_argument('--dev-ratio', type=float, default=0.2,
                        help='The fraction of points to be used as a development set.'
                             'Default is 0.2.')
    parser.add_argument('--sf', type=int, default=1,
                        help='A sampling factor for determining the number of '
                             'non-viral cascades. That is, k ('
                             'k = --sf * number-of-viral-cascades) non-viral cascades '
                             'will be sampled. Default is 1, equal samples of viral '
                             'and non-viral cascades are considered.')
    return parser.parse_args()


def main():
    print(argparse.__version__)
    # args = parser_args()
    # config = build_config(args)
    # cascades = load_cascades(config['cas_path'])
    # processed_cascades = process_observation_prediction_events(
    #     cascades=cascades,
    #     obs_time=config['observation_time'],
    #     prd_time=config['prediction_time'])
    # run_cas2vec(processed_cascades, config)


if __name__ == '__main__':
    main()
