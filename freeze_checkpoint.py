import tensorflow as tf
from tensorflow.contrib.slim.python.slim.nets import inception
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.framework import graph_util
slim = tf.contrib.slim

input_checkpoint = "/goshposh/Multi-label-Inception-net/openimages_2016_08/model.ckpt"
output_file = 'inference_graph.pb'


def PreprocessImage(image, central_fraction=0.875):
    """Load and preprocess an image.

    Args:
      image: a tf.string tensor with an JPEG-encoded image.
      central_fraction: do a central crop with the specified
        fraction of image covered.
    Returns:
      An ops.Tensor that produces the preprocessed image.
    """

    # Decode Jpeg data and convert to float.
    image = tf.cast(tf.image.decode_jpeg(image, channels=3), tf.float32)

    image = tf.image.central_crop(image, central_fraction=central_fraction)
    # Make into a 4D tensor by setting a 'batch size' of 1.
    image = tf.expand_dims(image, [0])
    image = tf.image.resize_bilinear(image,
                                     [299, 299],
                                     align_corners=False)

    # Center the image about 128.0 (which is done during training) and normalize.
    image = tf.multiply(image, 1.0 / 127.5)
    return tf.subtract(image, 1.0)

g = tf.Graph()
with g.as_default():
    input_image = tf.placeholder(tf.string, name='input')
    processed_image = PreprocessImage(input_image)

    with slim.arg_scope(inception.inception_v3_arg_scope()):
        logits, end_points = inception.inception_v3(processed_image, num_classes=6012, is_training=False)
        predictions = tf.nn.sigmoid(logits, name='multi_predictions')
        saver = tf_saver.Saver()
        input_graph_def = g.as_graph_def()
        sess = tf.Session()
        saver.restore(sess, input_checkpoint)

        output_node_names = "multi_predictions"
        output_graph_def = graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            input_graph_def, # The graph_def is used to retrieve the nodes
            output_node_names.split(",") # The output node names are used to select the usefull nodes
        )
        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_file, "wb") as f:
            f.write(output_graph_def.SerializeToString())