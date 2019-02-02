from pathlib import Path
from typing import Callable

import pytest
import tensorflow as tf
import numpy as np
import mnist

from elistimator.elistimator import Elistimator, TrainSpec, EvaluationSpec, PredictSpec


# <editor-fold desc="Fixtures">
@pytest.fixture(scope='module')
def mnist_dir(tmp_path_factory: Path):
    mnist_dir = Path(str(tmp_path_factory.mktemp('mnist_data')))
    if not mnist_dir.exists():
        mnist_dir.mkdir()

    mnist.temporary_dir = lambda: str(mnist_dir)

    return str(mnist_dir)


@pytest.fixture(scope='module')
def mnist_train(mnist_dir):
    mnist.temporary_dir = lambda: mnist_dir

    # Load train images and labels
    train_images = mnist.train_images()
    train_images = train_images.reshape(
        (train_images.shape[0], train_images.shape[1] * train_images.shape[2]))  # Flatten
    train_images = train_images.astype(np.float32)
    train_labels = mnist.train_labels()
    train_labels = train_labels.astype(np.int32)

    return train_images, train_labels


@pytest.fixture(scope='module')
def mnist_train_input_fn(mnist_train):
    return lambda: \
        tf.data.Dataset.from_tensor_slices({'images': mnist_train[0], 'labels': mnist_train[1]}) \
            .batch(56) \
            .prefetch(56).shuffle(buffer_size=1000)


@pytest.fixture(scope='module')
def mnist_test(mnist_dir):
    mnist.temporary_dir = lambda: mnist_dir

    # Load test images and labels
    test_images = mnist.test_images()
    test_images = test_images.reshape((test_images.shape[0], test_images.shape[1] * test_images.shape[2]))  # Flatten
    test_images = test_images.astype(np.float32)
    test_labels = mnist.test_labels()
    test_labels = test_labels.astype(np.int32)

    return test_images, test_labels


@pytest.fixture(scope='module')
def mnist_test_input_fn(mnist_test):
    return lambda: \
        tf.data.Dataset.from_tensor_slices({'images': mnist_test[0], 'labels': mnist_test[1]}) \
            .batch(56) \
            .prefetch(56)


@pytest.fixture(scope='function')
def trained_model_dir(tmp_path_factory: Path, mnist_train_input_fn: Callable, mnist_test_input_fn: Callable):
    model_dir = Path(str(tmp_path_factory.mktemp('trained_model')))
    if not model_dir.exists():
        model_dir.mkdir()

    # Create estimator
    estimator = Elistimator(model_fn=cnn_model_fn,
                            model_dir=str(model_dir),
                            params={
                                'img_height': 28,
                                'img_width': 28})

    # Setup
    estimator.setup(train_input_fn=mnist_train_input_fn)

    estimator.train().save_ckpt()

    return str(model_dir)


# </editor-fold>

# <editor-fold desc="Helpers">
def cnn_model_fn(features, is_training, params):
    input_layer = tf.reshape(features['images'], [-1, params['img_width'], params['img_height'], 1])

    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 28, 28, 1]
    # Output Tensor Shape: [batch_size, 28, 28, 32]
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 28, 28, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 14, 14, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 64]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 14, 14, 64]
    # Output Tensor Shape: [batch_size, 7, 7, 64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 7, 7, 64]
    # Output Tensor Shape: [batch_size, 7 * 7 * 64]
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 7 * 7 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=is_training)

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 10]
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    predict_spec = PredictSpec(output=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(features['labels'], depth=10), logits=logits)

    # Configure the Training Op (for TRAIN mode)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())

    train_spec = TrainSpec(loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=features['labels'], predictions=predictions["classes"])}

    eval_spec = EvaluationSpec(loss=loss, eval_metric_ops=eval_metric_ops)

    return train_spec, eval_spec, predict_spec


def cnn_model_pos_args_fn(features, is_training, params):
    input_layer = tf.reshape(features[0], [-1, params['img_width'], params['img_height'], 1])

    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 28, 28, 1]
    # Output Tensor Shape: [batch_size, 28, 28, 32]
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 28, 28, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 14, 14, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 64]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 14, 14, 64]
    # Output Tensor Shape: [batch_size, 7, 7, 64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 7, 7, 64]
    # Output Tensor Shape: [batch_size, 7 * 7 * 64]
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 7 * 7 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=is_training)

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 10]
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    predict_spec = PredictSpec(output=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(features[1], depth=10), logits=logits)

    # Configure the Training Op (for TRAIN mode)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())

    train_spec = TrainSpec(loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=features[1], predictions=predictions["classes"])}

    eval_spec = EvaluationSpec(loss=loss, eval_metric_ops=eval_metric_ops)

    return train_spec, eval_spec, predict_spec


# </editor-fold>

def test_train_only(tmpdir: str, mnist_train_input_fn: Callable):
    """
    Test train function
    """

    # Create model dir
    model_dir = Path(tmpdir + '/test_train_only')
    model_dir.mkdir()

    # Create estimator
    estimator = Elistimator(model_fn=cnn_model_fn,
                            model_dir=str(model_dir),
                            params={
                                'img_height': 28,
                                'img_width': 28})

    # Assert error
    with pytest.raises(RuntimeError) as excinfo:
        estimator.train()

    assert 'train method' in str(excinfo.value)

    # Setup
    estimator.setup(train_input_fn=mnist_train_input_fn)

    for _ in range(5):
        estimator.train().save_ckpt()

    # Assert error
    with pytest.raises(RuntimeError) as excinfo:
        estimator.validate()

    assert 'validate method' in str(excinfo.value)

    # Restore best
    estimator.restore_best_ckpt()
    estimator.restore_latest_ckpt()


def test_train_validate_ckpt(tmpdir: str, mnist_train_input_fn: Callable, mnist_test_input_fn: Callable):
    """
    Test train function with validation and checkpointing
    """

    # Create model dir
    model_dir = Path(tmpdir + '/test_train_validate')
    model_dir.mkdir()

    # Create estimator
    estimator = Elistimator(model_fn=cnn_model_fn,
                            model_dir=str(model_dir),
                            params={
                                'img_height': 28,
                                'img_width': 28})

    # Setup
    estimator.setup(train_input_fn=mnist_train_input_fn, validation_input_fn=mnist_test_input_fn)

    ckpt_global_steps = []
    ckpt_accuracies = []
    for _ in range(5):
        results = estimator.train().validate()
        estimator.save_ckpt()
        ckpt_global_steps.append(estimator.global_step)
        ckpt_accuracies.append(results['accuracy'])

    # Assert existence of all checkpoints
    assert len(estimator.checkpoints) == 5
    for _, ckpt in estimator.checkpoints.items():
        assert Path(ckpt['file'] + '.meta').exists()

    # Test checkpoint restoring
    estimator.restore_ckpt(path_prefix=estimator.checkpoints[ckpt_global_steps[2]]['file'])

    assert estimator.validate()['accuracy'] == ckpt_accuracies[2]

    # Restore best
    estimator.restore_best_ckpt()

    assert estimator.validate()['accuracy'] == ckpt_accuracies[-1]

    # Restore latest
    estimator.restore_ckpt(path_prefix=estimator.checkpoints[ckpt_global_steps[2]]['file'])
    assert estimator.validate()['accuracy'] == ckpt_accuracies[2]
    estimator.restore_latest_ckpt()

    assert estimator.validate()['accuracy'] == ckpt_accuracies[-1]


def test_recovery(trained_model_dir: str, mnist_train_input_fn: Callable, mnist_test_input_fn: Callable):
    """
    Test recovery and continuing training from persisted directory
    """

    estimator = Elistimator.from_disk(model_dir=trained_model_dir, from_best=True)

    # Assert error
    with pytest.raises(RuntimeError) as excinfo:
        estimator.train()

    assert 'train method' in str(excinfo.value)

    # Setup
    estimator.setup(train_input_fn=mnist_train_input_fn, validation_input_fn=mnist_test_input_fn)

    estimator.train().validate()


def test_predict(trained_model_dir: str, mnist_test_input_fn: Callable, mnist_test):
    """
    Test the predict method functionality
    """

    estimator = Elistimator.from_disk(model_dir=trained_model_dir)

    # Using input function
    for p in estimator.predict(input_fn=mnist_test_input_fn):
        pass

    # Using one off
    image = mnist_test[0][:1]
    estimator.predict(images=image)


def test_evaluate(trained_model_dir: str, mnist_test_input_fn: Callable):
    """
    Test the evaluate method functionality
    """

    estimator = Elistimator.from_disk(model_dir=trained_model_dir)

    # Using input function
    estimator.evaluate(input_fn=mnist_test_input_fn)


def test_positional_arguments(tmpdir, mnist_train, mnist_test):
    """
    Test full functionality using positional arguments
    """

    train_images, train_labels = mnist_train
    test_images, test_labels = mnist_test

    def example_generator(images, labels):
        for image, label in zip(images, labels):
            yield image, label

    train_fn = lambda: \
        tf.data.Dataset.from_generator(lambda: example_generator(train_images, train_labels),
                                       output_types=(tf.float32, tf.int32)).batch(56).prefetch(56)

    test_fn = lambda: \
        tf.data.Dataset.from_generator(lambda: example_generator(test_images, test_labels),
                                       output_types=(tf.float32, tf.int32)).batch(56).prefetch(56)

    # Create model dir
    model_dir = Path(tmpdir + '/test_positional_arguments')
    model_dir.mkdir()

    # Create estimator
    estimator = Elistimator(model_fn=cnn_model_pos_args_fn,
                            model_dir=str(model_dir),
                            params={
                                'img_height': 28,
                                'img_width': 28})

    # Setup
    estimator.setup(train_input_fn=train_fn, validation_input_fn=test_fn)

    ckpt_global_steps = []
    ckpt_accuracies = []
    for _ in range(5):
        results = estimator.train().validate()
        estimator.save_ckpt()
        ckpt_global_steps.append(estimator.global_step)
        ckpt_accuracies.append(results['accuracy'])

    # Assert existence of all checkpoints
    assert len(estimator.checkpoints) == 5
    for _, ckpt in estimator.checkpoints.items():
        assert Path(ckpt['file'] + '.meta').exists()

    # Test checkpoint restoring
    estimator.restore_ckpt(path_prefix=estimator.checkpoints[ckpt_global_steps[2]]['file'])

    assert estimator.validate()['accuracy'] == ckpt_accuracies[2]

    # Restore best
    estimator.restore_best_ckpt()

    assert estimator.validate()['accuracy'] == ckpt_accuracies[-1]

    # Restore latest
    estimator.restore_ckpt(path_prefix=estimator.checkpoints[ckpt_global_steps[2]]['file'])
    assert estimator.validate()['accuracy'] == ckpt_accuracies[2]
    estimator.restore_latest_ckpt()

    assert estimator.validate()['accuracy'] == ckpt_accuracies[-1]

    estimator.predict(train_images[:10])
