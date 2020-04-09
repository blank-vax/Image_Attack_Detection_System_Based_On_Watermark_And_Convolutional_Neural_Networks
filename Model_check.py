import tensorflow as tf
import numpy as np
import matlab.engine
from PIL import Image
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

def eval():
    # Model path
    CHECKPOINT_DIR = 'checkpoints' # The store location of the model after training
    INCEPTION_MODEL_FILE = 'tensorflow_inception_graph.pb' # The store location of the inception-v3 model 

    # The parameters of the inception-v3 model
    BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'  # The name of tensor representing the result of bottleneck layer in inception-v3 model

    JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'  # The corresponding name of the tensor of input image

    # Test data
    file_path = 'output\\out.jpg'

    # Read data
    image_data = tf.gfile.GFile(file_path, 'rb').read()

    # Evaluate
    checkpoint_file = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    with tf.Graph().as_default() as graph:
        with tf.Session().as_default() as sess:
            # Open the trained inception-v3 model
            with tf.gfile.GFile(INCEPTION_MODEL_FILE, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
            # Load inception-v3 model, and return the tensors of data input and bottleneck layer output  

            bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(
                graph_def,
                return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME])
            # Using inception-v3 model to process the picture to get character vectors
            bottleneck_values = sess.run(bottleneck_tensor,
                                        {jpeg_data_tensor: image_data})
            
            # Compress four-dimensional matrix to one-dimensional matrix
            bottleneck_values = [np.squeeze(bottleneck_values)]
            # Load meta graph and variables
            saver = tf.train.import_meta_graph('{}.meta'.format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the enter placeholder from pictures by name
            input_x = graph.get_operation_by_name(
                'BottleneckInputPlaceholder').outputs[0]

            # The tensors we wanna to evaluate
            predictions = graph.get_operation_by_name('evaluation/ArgMax').outputs[
                0]

            # Collect the predicting value
            all_predictions = []
            all_predictions = sess.run(predictions, {input_x: bottleneck_values})

    # If we get the labels then print the accuracy rate

    if True:
        print("\nDetect result：")
        if all_predictions == [0]:
            print("This picture has been cut out\n")
        if all_predictions == [1]:
            print("This picture has been processed by filters\n")
        if all_predictions == [2]:
            print("This picture has been scaled\n")
        if all_predictions == [3]:
            print("This picture has been daubed\n")
        if all_predictions == [4]:
            print("This picture is normal\n")


def IsValidImage(img_path):
    """
    Tell the effectiveness of the picture
    :param img_path: image path
    """
    bValid = True
    try:
        Image.open(img_path).verify()
    except:
        bValid = False
    return bValid


def transimg(img_path):
    """
    Picture format transform: png to jpg
    :param img_path: image path
    """
    if IsValidImage(img_path):
        try:
            str = img_path.rsplit(".", 1)
            output_img_path = str[0] + ".jpg"
            im = Image.open(img_path)
            im.save(output_img_path)
            return True
        except:
            return False
    else:
        return False


def main():
    print("Start, please wait...")
    eng = matlab.engine.start_matlab()
    while True:
        filepath1 = input("Please input the picture needing detect: ")
        filepath2 = "output\\out.png"
        # Extract watermark from the input picture
        eng.extract(filepath1, filepath2, nargout = 0)
        filepath3 = "output\\out.png"
        # Transform the png to jpg
        transimg(filepath3)
        print("\nWatermark extraction finished!Detection start, please wait...")
        eval()
        print("\nChoose next step：\n    1.Continue to detect\n    2.Exit")
        flag = input("Please input your choice：")
        if (flag == '2'):
            eng.quit()
            exit()
        elif (flag != '1'):
            print("Wrong parameters! Exiting...")
            eng.quit()
            exit()
        os.system('cls')


if __name__ == "__main__":
    main()