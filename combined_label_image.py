import tensorflow as tf
import sys
import os
from tensorflow.contrib import learn
import numpy as np
import re
import urllib

# change this as you see fit
image_path = "http://dtpmhvbsmffsz.cloudfront.net/posts/2014/09/17/541a5558b539e42fdf0bcfc4/m_541a555eb539e42fdf0bcfce.jpg"

req = urllib.request.Request(image_path)
response = urllib.request.urlopen(req)
image_data = response.read()

test_description = " Peacock inspired Necklace NWOT  Beautiful Peacock color inspired necklace. Never worn! Price negotiable  use the the offer button"

# Read in the image_data
#image_data = tf.gfile.FastGFile(image_path, 'rb').read()

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile("labels.txt")]

def clean_str(s):
    """Clean sentence"""
    s = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", s)
    s = re.sub(r"\'s", " \'s", s)
    s = re.sub(r"\'ve", " \'ve", s)
    s = re.sub(r"n\'t", " n\'t", s)
    s = re.sub(r"\'re", " \'re", s)
    s = re.sub(r"\'d", " \'d", s)
    s = re.sub(r"\'ll", " \'ll", s)
    s = re.sub(r",", " , ", s)
    s = re.sub(r"!", " ! ", s)
    s = re.sub(r"\(", " \( ", s)
    s = re.sub(r"\)", " \) ", s)
    s = re.sub(r"\?", " \? ", s)
    s = re.sub(r"\s{2,}", " ", s)
    s = re.sub(r'\S*(x{2,}|X{2,})\S*', "xxx", s)
    s = re.sub(r'[^\x00-\x7F]+', "", s)
    return s.strip().lower()

# Unpersists graph from file
with tf.gfile.FastGFile("/goshposh/Multi-label-Inception-net/models/combined/combined_output_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    vocab_path = os.path.join('/goshposh/Multi-label-Inception-net/models/combined/', "vocab.pickle")
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)

    description = np.array(list(vocab_processor.transform(clean_str(test_description))))

    description_tensor = sess.graph.get_operation_by_name("input/DescriptionsInput").outputs[0]

    # Feed the image_data as input to the graph and get first prediction
    sigmoid_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
    predictions = sess.run(sigmoid_tensor, \
             {'DecodeJpeg/contents:0': image_data,
              description_tensor: description})
    
    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        print('%s (score = %.5f)' % (human_string, score))
    

    filename = "results.txt"    
    with open(filename, 'a+') as f:
        f.write('\n**%s**\n' % (image_path))
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            f.write('%s (score = %.5f)\n' % (human_string, score))
    
    
