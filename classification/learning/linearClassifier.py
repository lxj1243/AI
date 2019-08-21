import tensorflow as tf

# set up a linear classifier
classifier = tf.estimator.LinearClassifier()

# train the model on some example data
classifier.train(input_fn=train_input_fn, steps=2000)

# use it to predict
classifier.predict(input_fn=predict_input_fn)
