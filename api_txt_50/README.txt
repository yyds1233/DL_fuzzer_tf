TensorFlow 50 API txt bundle

Count: 50

Filename rule:
- every "." in the API name is replaced with "_"
- example: tf.nn.conv2d -> tf_nn_conv2d.txt

Content rule:
- first line is the original API name
- includes official TensorFlow documentation URL
- normalized into parser-friendly sections: Shape / Input / Output / Constraints

This bundle is intended for shape/rank/constraint extraction.
