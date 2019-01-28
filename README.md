# tf-apis
Sample scripts to demonstrate use of tf APIs

## TF-Estimator API

### Structure of a pre-made Estimators program

#### Write one or more dataset importing functions

```python
def input_fn(dataset):
    """
    feature_dict = keys are feature names and values are tensors
    label = a tensor containing one or more labels
    """
    ... # manipulate dataset, extracting the feature dict and the label
    return feature_dict, label
```

#### Define the feature columns
```python
# Define three numeric feature columns.
population = tf.feature_column.numeric_column('population')
crime_rate = tf.feature_column.numeric_column('crime_rate')
median_education = tf.feature_column.numeric_column('median_education',
                    normalizer_fn=lambda x: x - global_education_mean)
```

#### Instantiate the relevant pre-made estimator
```python
# Instantiate an estimator, passing the feature columns.
estimator = tf.estimator.LinearClassifier(
    feature_columns=[population, crime_rate, median_education],
    )
```

#### Call a training, evaluation, or inference method

```python
# my_training_set is the function created in Step 1
estimator.train(input_fn=my_training_set, steps=2000)
```

#### Creating Estimators from Keras models

```python
# Instantiate a Keras inception v3 model.
keras_model = tf.keras.applications.inception_v3.InceptionV3(weights=None)
# Compile model with the optimizer, loss, and metrics you'd like to train with.
keras_model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9),
                          loss='categorical_crossentropy',
                          metric='accuracy')
# Create an Estimator from the compiled Keras model. Note the initial model
# state of the keras model is preserved in the created Estimator.
est_model = tf.keras.estimator.model_to_estimator(keras_model=keras_model)

# Treat the derived Estimator as you would with any other Estimator.
# First, recover the input name(s) of Keras model, so we can use them as the
# feature column name(s) of the Estimator input function:
keras_model.input_names  # print out: ['input_1']
# Once we have the input name(s), we can create the input function, for example,
# for input(s) in the format of numpy ndarray:
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"input_1": train_data},
    y=train_labels,
    num_epochs=1,
    shuffle=False)
# To train, we call Estimator's train function:
est_model.train(input_fn=train_input_fn, steps=2000)
```