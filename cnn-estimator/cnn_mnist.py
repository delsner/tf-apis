"""
Run with

python cnn_mnist.py --training_steps=10 --learning_rate=0.001 --batch_size=128

"""

from __future__ import absolute_import, print_function, division

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_float(
    name='learning_rate',
    default=0.001,
    help='Learning rate.')
tf.flags.DEFINE_string(
    name='model_dir',
    default='run',
    help='Output directory for model and training stats.')
tf.flags.DEFINE_integer(
    name='batch_size',
    default=128,
    help='Batch size used for training.')
tf.flags.DEFINE_integer(
    name='training_steps',
    default=20000,
    help='Amount of training steps.')


def model_fn(features, labels, mode, config):
    """
    Model function conforming with interface of TF Estimator API

    Returns `EstimatorSpec` which fully defines the model to be run by an `Estimator`.
    :param features: X
    :param labels: y
    :param mode: [TRAIN, EVAL, PREDICT] (see tf.estimator.ModeKeys)
    :return: training_op, loss (for evaluation metrics), predictions
    """
    # input layer
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])  # shape = (batch_size, 28, 28, 1)

    # 1st conv2d + maxpooling2d, reduces to (-1, 14, 14, 32) -> 32 = filter_size
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu
    )
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[2, 2],
        strides=2
    )

    # 2nd conv2d + maxpooling2d, reduces to (-1, 7, 7, 64) -> 64 = filter_size
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[2, 2],
        strides=2)

    # dense layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(
        inputs=pool2_flat,
        units=1024,
        activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense,
        rate=0.4,
        training=mode == tf.estimator.ModeKeys.TRAIN)  # necessary to rescale on dev/test

    # logits layer
    logits = tf.layers.dense(
        inputs=dropout,
        units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=labels,
        logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=FLAGS.learning_rate)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels,
            predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # set custom checkpointing config
    checkpointing_config = tf.estimator.RunConfig(
        save_checkpoints_secs=10 * 60,  # Save checkpoints every 20 minutes.
        keep_checkpoint_max=10,  # Retain the 10 most recent checkpoints.
    )

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=FLAGS.model_dir,
        config=checkpointing_config
    )

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log,
        every_n_iter=50)

    # Define input fn
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=FLAGS.batch_size,
        num_epochs=None,
        shuffle=True)

    # Train classifier
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=FLAGS.training_steps,
        hooks=[logging_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(
        input_fn=eval_input_fn)
    print(eval_results)


if __name__ == '__main__':
    tf.app.run()
