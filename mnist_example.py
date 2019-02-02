from elistimator.elistimator import Elistimator, TrainSpec, EvaluationSpec, PredictSpec
import tensorflow as tf  # pip install tensorflow_gpu
import mnist  # pip install mnist
import numpy as np  # pip install numpy

# Load train images and labels
train_images = mnist.train_images()
train_images = train_images.reshape(
    (train_images.shape[0], train_images.shape[1] * train_images.shape[2]))  # Flatten
train_images = train_images.astype(np.float32)
train_labels = mnist.train_labels()
train_labels = train_labels.astype(np.int32)

# Load test images and labels
test_images = mnist.test_images()
test_images = test_images.reshape((test_images.shape[0], test_images.shape[1] * test_images.shape[2]))  # Flatten
test_images = test_images.astype(np.float32)
test_labels = mnist.test_labels()
test_labels = test_labels.astype(np.int32)

params = {
    'img_height': 28,
    'img_width': 28,
    'batch_size': 56,
    'epochs': 5}

# Input functions
train_input_fn = lambda: \
    tf.data.Dataset.from_tensor_slices({'images': train_images, 'labels': train_labels}) \
        .batch(params['batch_size']) \
        .prefetch(params['batch_size']).shuffle(buffer_size=1000)

validation_input_fn = lambda: \
    tf.data.Dataset.from_tensor_slices({'images': test_images, 'labels': test_labels}) \
        .batch(params['batch_size']) \
        .prefetch(params['batch_size'])


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
        # Generate predictions (for PredictSpec and EvaluationSpec)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PredictSpec.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    predict_spec = PredictSpec(output=predictions)

    # Calculate Loss (for both TrainSpec and EvaluationSpec)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(features['labels'], depth=10), logits=logits)

    # Configure the Training Op (for TrainSpec)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())

    train_spec = TrainSpec(loss=loss, train_op=train_op)

    # Add evaluation metrics (for EvaluationSpec)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=features['labels'], predictions=predictions["classes"])}

    eval_spec = EvaluationSpec(loss=loss, eval_metric_ops=eval_metric_ops)

    return train_spec, eval_spec, predict_spec


# 50% GPU restriction
session_args = {'config': tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5))}

# Create the Elistimator instance
estimator = Elistimator(model_fn=cnn_model_fn,
                        params=params,
                        session_args=session_args)

# Setup
estimator.setup(train_input_fn=train_input_fn,
                validation_input_fn=validation_input_fn,
                train_saver_args={'max_to_keep': 3})  # Save maximum of 3 checkpoints

# Train and validation loop
for _ in range(params['epochs']):
    estimator.train().validate()
    estimator.save_ckpt()

# Predicting using an input function
all_predictions = []
for p in estimator.predict(input_fn=validation_input_fn):
    all_predictions.append(p)


# Predicting using one-off predictions
prediction = estimator.predict(images=test_images[:1])  # Predict single image


# Evaluating
evaluation = estimator.evaluate(input_fn=validation_input_fn)

