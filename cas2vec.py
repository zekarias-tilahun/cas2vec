from cas2vec_helper import *
import tensorflow as tf

import numpy as np


def _build_counter_sequence(events, bins):
    """
    Builds a counter sequence for specified series of events based on given
    bins (originally called slices)
    :param events: The series of events
    :param bins: The bins (slices)
    :return: list: counter sequence
    """
    counter_sequence = []
    last_pos = 0
    for i in range(bins.shape[0]):
        timestamps = events[last_pos:]
        event_count = trace(timestamps, endtime=bins[i]).shape[0]
        if event_count > 0:
            event_count = int(ceilk(event_count, 10))
        counter_sequence.append(event_count)
        last_pos = last_pos + event_count

    return counter_sequence


def _build_constant_sequence(events, bins):
    """
    Builds a constant (originally called discrete) sequence for specified
    series of events based on given bins (originally called slices)
    :param events: The series of events
    :param bins: The bins (slices)
    :return: list: constant sequence
    """
    constant_sequence = []
    last_pos = 0
    for i in range(bins.shape[0]):
        timestamps = events[last_pos:]
        event_count = trace(timestamps, endtime=bins[i]).shape[0]
        if event_count > 0:
            entry = [i + 1] * event_count
            constant_sequence = constant_sequence + entry
        last_pos = last_pos + event_count

    return constant_sequence


def cas2vec_transform(data, bins, disc_method, sequence_length):
    """
    Transform diffusion event data into discrete valued sequences as described
    in the cas2vec paper by using bins (slices)
    :param data: Cascade data
    :param bins: Bins (slices)
    :param disc_method: discretization method
    :param sequence_length The sequence length
    :return:
    """
    print('Transforming cascades according to the {} discretization method'.format(disc_method))
    if disc_method == 'counter':
        sequence_builder = _build_counter_sequence
    else:
        sequence_builder = _build_constant_sequence

    output = {'cid': [], 'sequence': [], 'label': []}
    for cid, _, events_at_o, num_prd_events in data:
        counter_sequence = sequence_builder(events_at_o, bins)
        output['cid'].append(cid)
        output['sequence'].append(counter_sequence)
        output['label'].append(num_prd_events)
    output['sequence'] = np.array(output['sequence'])
    if disc_method == 'const':
        output['sequence'] = tf.keras.preprocessing.sequence.pad_sequences(
            sequences=output['sequence'], padding='post',
            maxlen=sequence_length)
    output['label'] = np.array(output['label'])
    return output


def build_cnn_input(config):
    """
    Builds an input and embedding layer according to a configuration
    :param config: The configuration
    :return:
    """
    input_layer = tf.keras.layers.Input(
        shape=(config['sequence_length'],), name='input_sequence')
    embedding_layer = tf.keras.layers.Embedding(
        input_dim=len(config['vocab']), output_dim=config['emb_size'],
        input_length=config['sequence_length'],
        name='embedding_layer')(input_layer)
    return input_layer, embedding_layer


def build_cnn_layer(embedding_layer, config):
    """
    Builds a 1D CNN layer with max pooling according to a configuration
    :param embedding_layer: Embedding layer
    :param config: The configuration
    :return:
    """
    layers = []
    for i in range(len(config['filters'])):
        cnn_layer = tf.keras.layers.Conv1D(
            kernel_regularizer=tf.keras.regularizers.l2(0.0001),
            activation='relu', filters=config['filters'][i],
            kernel_size=config['kernel_size'][i],
            name='cnn_layer_{}'.format(i + 1))(embedding_layer)
        pooling_layer = tf.keras.layers.MaxPool1D(
            name='max_pool_layer_{}'.format(i + 1))(cnn_layer)
        flatten_layer = tf.keras.layers.Flatten(
            name='flatten_{}'.format(i + 1))(pooling_layer)
        layers.append(flatten_layer)
    return tf.keras.layers.Concatenate(name='concat_layer')(layers)


def build_mlp(input_, config):
    """
    Builds a multi-layer-perceptron according to a configuration
    :param input_: The input the mlp
    :param config: The configuration
    :return:
    """
    current_input = input_
    for i in range(len(config['fcc_layers']) - 1):
        current_input = tf.keras.layers.Dense(
            units=config['fcc_layers'][i], activation='tanh',
            name='fcc_layer_{}'.format(i + 1))(current_input)
    cascade_embedding_layer = tf.keras.layers.Dense(
        units=config['fcc_layers'][-1], activation='tanh',
        name='cascade_embedding_layer')(current_input)
    prediction_layer = tf.keras.layers.Dense(
        units=1, activation='sigmoid',
        name='prediction_layer')(cascade_embedding_layer)
    return cascade_embedding_layer, prediction_layer


def cas2vec_model(config):
    """
    Builds a cas2vec model according to a configuration
    :param config:
    :return:
    """
    input_layer, embedding_layer = build_cnn_input(config)
    fcc_layer = build_cnn_layer(embedding_layer, config)
    cascade_embedding_layer, prediction_layer = build_mlp(fcc_layer, config)
    model = tf.keras.models.Model(
        inputs=input_layer, outputs=prediction_layer)
    cascade_embedding_predictor = tf.keras.models.Model(
        inputs=input_layer, outputs=prediction_layer)

    model.compile(
        optimizer=tf.train.AdamOptimizer(config['learning_rate']),
        loss='binary_crossentropy', metrics=config['metrics'])

    return {
        'model': model,
        'cascade_embedding_predictor':cascade_embedding_predictor
    }


def run_cas2vec(processed_cascades, config):
    model_config = cas2vec_model(config)
    print(model_config['model'].summary())
    transformed_cascades = cas2vec_transform(
        processed_cascades, config['bins'], disc_method=config['disc_method'],
        sequence_length=config['sequence_length'])
    labels = binarize_labels(labels=np.array(transformed_cascades['label']))
    sampling_results = sample(
        labels=labels, sequence_data=transformed_cascades['sequence'],
        factor=config['sampling_factor'])
    indices = list(range(sampling_results['sequence'].shape[0]))
    random.shuffle(indices)
    sequence_inputs = sampling_results['sequence'][indices]
    label_inputs = sampling_results['label'][indices]
    model_config['model'].fit(
        x=sequence_inputs, y=label_inputs, validation_split=config['dev_ratio'], 
        epochs=config['epochs'], batch_size=config['batch_size'])








