# Use a feedforward neural network on the iris dataset
# Data import codes partially from the original tutorial

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import pandas as pd
import tensorflow as tf
import numpy as np

TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')
args = parser.parse_args([])

def maybe_download():
    train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)
    test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)

    return train_path, test_path

train_path, test_path = maybe_download()

y_name='Species'
train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
train_x, train_y = train, train.pop(y_name)

test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
test_x, test_y = test, test.pop(y_name)
train_x

# n_layer feedforward neural network
# this is to test the feedforward neural network if it works (CS294 HW2)
# the approach followes the MNIST tutorial
def build_mlp(
        input_placeholder, 
        output_size,
        scope, 
        n_layers=2, 
        size=64, 
        activation=tf.tanh,
        output_activation=None
        ):
    with tf.variable_scope(scope):
        # YOUR_CODE_HERE
        # Define Input layer
        input_layer=input_placeholder
        # Define dense layers
        dense=tf.layers.dense(inputs=input_layer, units=size, activation = activation)
        for i in range(n_layers-1):
            dense=tf.layers.dense(inputs=dense, units=size, activation=activation)
        # Define output layer
        output_layer=tf.layers.dense(inputs=dense, units=output_size, activation=output_activation)
        return output_layer

# Define the classifier using the feedforward neural network
def ffmodel_fn(features, labels, mode):
    logits = build_mlp(input_placeholder=features, output_size=3, scope="irisclassify")
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    # To simulate the DNNclassifier, use losses.Reduction.SUM
    #loss = tf.losses.SUM #not working for some reason
    
    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.03) #MNIST set the learning rate as 0.001 
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    
    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
    #prepare input_layer
    train_labels = np.asarray(train_y, dtype=np.int32)
    train_x_np= train_x.values
    train_x_np = train_x_np.astype(np.float32, copy=False)
    train_y_np = train_y.values
    train_y_np = train_y_np.astype(np.int32, copy=False)
    train_input_irisfn= tf.estimator.inputs.numpy_input_fn(
            x=train_x_np,
            y=train_y_np,
            batch_size=args.batch_size,
            num_epochs=None,
            shuffle=True)
    # Set up logging for prediction
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=50)
    iris_classifier = tf.estimator.Estimator(model_fn=ffmodel_fn, 
            model_dir="/tmp/iris_ffn_model")
    #Train the classifier
    iris_classifier.train(
            input_fn=train_input_irisfn,
            steps=args.train_steps,
            hooks=[logging_hook])
    #Evaluate the trained classifier
    test_x_np = test_x.values
    test_x_np = test_x_np.astype(np.float32, copy=False)
    test_y_np = test_y.values
    test_y_np = test_y_np.astype(np.int32, copy=False)
    print(type(test_x_np))
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x=test_x_np,
            y=test_y_np,
            num_epochs=1,
            shuffle=False)
    eval_results = iris_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)
    # Generate predictions from the model
    expected = ['Setosa', 'Versicolor', 'Virginica']
    predict_x = {
            'SepalLength': [5.1, 5.9, 6.9],
            'SepalWidth': [3.3, 3.0, 3.1],
            'PetalLength': [1.7, 4.2, 5.4],
            'PetalWidth': [0.5, 1.5, 2.1],
            }
    predict_x = [[5.1, 3.3, 1.7, 0.5], [5.9,3.0,4.2,1.5],[6.9,3.1,5.4,2.1]]
    predict_x = np.array(predict_x).astype(np.float32, copy=False)
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x = predict_x,
            #y = None
            num_epochs = 1,
            shuffle = False
            )
    predict_results = list(iris_classifier.predict(input_fn=predict_input_fn))
    template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')
    
    for pred_dict, expec in zip(predict_results, expected):
        class_id = pred_dict['classes']
        probability = pred_dict['probabilities'][class_id]
        
        print(template.format(SPECIES[class_id],
                              100 * probability, expec))

if __name__ == "__main__":
  tf.app.run()
