import tensorflow as tf
import numpy as np
import random


def load_cascades(path):
    print('Reading data from {}'.format(path))
    cascades = []
    with open(path, 'r') as f:
        for line in f:
            ln = line.strip().split()
            c = []
            users = []
            cid = int(ln[0])
            for event in ln[1:]:
                e = event.strip().split(',')
                ts = float(e[1])
                u = float(e[0])
                users.append(u)
                c.append(ts)

            users = np.array(users)
            c = np.array(c)
            order = np.argsort(c)
            cascades.append([cid, users[order], c[order]])
    return cascades


def process_observation_prediction_events(cascades, obs_time, prd_time):
    print('Processing observation and prediction events ...')
    data = []
    for cid, structural_info, timestamps in cascades:
        events_at_o = trace(timestamps, 0, obs_time)
        num_obs_events = events_at_o.shape[0]
        if num_obs_events > 4:
            structural_info_at_o = structural_info[: num_obs_events]
            events_at_p = trace(timestamps, 0, prd_time)
            num_prd_events = events_at_p.shape[0]
            data.append((cid, structural_info_at_o, events_at_o, num_prd_events))
    return data


def time_conversion_factor(time_units):
    if time_units == 's':
        return 1.
    elif time_units == 'h':
        return 3600.
    elif time_units == 'm':
        return 60.


def build_bins(args):
    tcf = time_conversion_factor(time_units=args.time_unit)
    bin_size = args.obs_time / args.num_bins
    bins = np.arange(bin_size, args.obs_time + bin_size, bin_size)
    return bins * tcf


def build_vocab(args, round_factor=10):
    print('Building vocabulary')
    if args.disc_method == 'counter':
        max_val = 50000
        step = round_factor
    else:
        max_val = args.num_bins
        step = 1

    vocab = list(range(0, max_val, step))
    print('Number of items in vocabulary: {}'.format(len(vocab)))
    return dict(zip(vocab, range(len(vocab))))


def floork(num, k=100):
    return np.array(num) // k * k


def ceilk(num, k=100):
    return np.array(num) // k * k + k


def trace(c, starttime=None, endtime=None):
    if starttime is None:
        return c[c < endtime]
    elif endtime is None:
        return c[c >= starttime]

    return c[(c >= starttime) & (c < endtime)]


def select_and_concat(data, index1, index2, order=None):
    """
    Selects two sets of data points based on two sets of indices
    and concatenate the selected points and arranges them in a
    given random order
    :param data: The full data from which data points will be selected
    :param index1: The first set of indices
    :param index2: The second set of indices
    :param order: The order to arrange the selected and combined data points
    :return:
    """
    concat = np.concatenate([data[index1], data[index2]])
    if order is not None:
        return concat[order]
    return concat


def build_config(args):
    bins = build_bins(args)
    vocab = build_vocab(args=args)
    tcf = time_conversion_factor(args.time_unit)
    config = {
        'cas_path': args.cas_path, 'model_dir': args.model_dir,
        'time_unit': args.time_unit, 'num_bins': args.num_bins,
        'observation_time': args.obs_time * tcf,
        'prediction_time': args.prd_time * tcf,
        'sequence_length': args.seq_len if args.disc_method == 'const' else args.num_bins,
        'threshold': args.threshold,
        'dev_ratio': args.dev_ratio, 'sampling_factor': args.sf,
        'filters': [32, 16] if args.filters is None else args.filters,
        'kernel_size': [3, 4] if args.kernel_size is None else args.kernel_size,
        'fcc_layers': [512, 128] if args.fcc_layers is None else args.fcc_layers,
        'bins': bins, 'disc_method': args.disc_method,
        'vocab': vocab, 'emb_size': args.size, 'learning_rate': args.lr,
        'metrics': [precision, recall, fmeasure],
        'epochs': args.epochs, 'batch_size': args.batch_size
    }
    return config


def pretty_display(args):
    for arg in vars(args):
        print('{}: {}'.format(arg, getattr(args, arg)))


def binarize_labels(labels, threshold=1000):
    """
    Binarize real valued labels as viral(1) and non-viral(0) according to a
    specified threshold
    :param labels: The real valued labels
    :param threshold: The threshold
    :return:
    """
    print('Changing real label to binary at threshold: {}'.format(threshold))
    binary_labels = np.zeros(shape=labels.shape)
    nnz = np.argwhere(labels > threshold).reshape(-1)
    binary_labels[nnz] = 1
    return binary_labels


def sample(labels, sequence_data=None, factor=2):
    """
    Sample non-viral points that is directly proportional to the number of
    viral points by a specified factor
    :param labels: Binary labels
    :param sequence_data: A sequence data
    :param factor: The factor
    :return:
    """

    print('Sampling viral and non-viral cascades')
    ones = np.argwhere(labels > 0).reshape(-1)
    zeros = [i for i in range(labels.shape[0]) if i not in ones]
    sample_zeros = random.sample(zeros, factor * ones.shape[0])
    rand_positions = list(range(ones.size + len(sample_zeros)))
    random.shuffle(rand_positions)
    sampled_labels = select_and_concat(labels, ones, sample_zeros, order=rand_positions)
    output = {'label': sampled_labels}
    sampled_sequences = select_and_concat(sequence_data, ones, sample_zeros, order=rand_positions)
    output['sequence'] = sampled_sequences

    print('Sample sizes')
    print('Viral cascades: {}'.format(ones.size))
    print('Non-viral cascades: {}'.format(len(sample_zeros)))

    return output


def recall(y_true, y_hat):
    """
    Taken from https://github.com/keras-team/keras/commit/a56b1a55182acf061b1eb2e2c86b48193a0e88f7
    Recall metric.

    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_hat, 0, 1)))
    possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
    recall_ = true_positives / (possible_positives + tf.keras.backend.epsilon())
    return recall_


def precision(y_true, y_hat):
    """
    Taken from https://github.com/keras-team/keras/commit/a56b1a55182acf061b1eb2e2c86b48193a0e88f7

    Precision metric.

    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_hat, 0, 1)))
    predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_hat, 0, 1)))
    precision_ = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    return precision_


def _fbeta_score(y_true, y_hat, beta=1):
    """
    Taken from https://github.com/keras-team/keras/commit/a56b1a55182acf061b1eb2e2c86b48193a0e88f7

    Computes the F score.

    The F score is the weighted harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.

    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.

    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    """
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_hat)
    r = recall(y_true, y_hat)
    bb = beta ** 2
    fbeta_score_ = (1 + bb) * (p * r) / (bb * p + r + tf.keras.backend.epsilon())
    return fbeta_score_


def fmeasure(y_true, y_hat, beta=3):
    """
    Taken from https://github.com/keras-team/keras/commit/a56b1a55182acf061b1eb2e2c86b48193a0e88f7

    Computes the f-measure, the harmonic mean of precision and recall.

    Here it is only computed as a batch-wise average, not globally.
    """
    return _fbeta_score(y_true, y_hat, beta=beta)
