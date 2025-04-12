import numpy as np
import tensorflow.compat.v1 as tf
import time
tf.disable_v2_behavior()

def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.compat.v1.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def, name="")

    return graph

def read_tensor_from_image_file(file_name, input_height=299, input_width=299, input_mean=128, input_std=128):
    file_reader = tf.io.read_file(file_name)
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(file_reader, channels=3)
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(tf.image.decode_gif(file_reader))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader)
    else:
        image_reader = tf.image.decode_jpeg(file_reader, channels=3)

    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    
    with tf.Session() as sess:
        result = sess.run(normalized)
    return result

def load_labels(label_file):
    label = []
    with open(label_file, "r") as f:
        label = [line.strip() for line in f.readlines()]
    return label

def main(img_path):
    model_file = "retrained_graph.pb"
    label_file = "retrained_labels.txt"
    input_height = 299
    input_width = 299
    input_mean = 128
    input_std = 128
    input_layer = "Mul"
    output_layer = "final_result"

    graph = load_graph(model_file)
    t = read_tensor_from_image_file(
        img_path,
        input_height=input_height,
        input_width=input_width,
        input_mean=input_mean,
        input_std=input_std,
    )

    input_operation = graph.get_operation_by_name(input_layer)
    output_operation = graph.get_operation_by_name(output_layer)

    with tf.Session(graph=graph) as sess:
        start = time.time()
        results = sess.run(output_operation.outputs[0], {input_operation.outputs[0]: t})
        end = time.time()

    results = np.squeeze(results)
    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(label_file)

    # Return the top prediction
    top_prediction = labels[top_k[0]]
    confidence = results[top_k[0]]
    return f"{top_prediction} ({confidence * 100:.2f}% confidence)"
