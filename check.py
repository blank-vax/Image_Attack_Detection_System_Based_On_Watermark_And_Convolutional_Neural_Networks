import tensorflow as tf
import numpy as np
import matlab.engine
from PIL import Image
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

def eval():
    # 模型目录
    CHECKPOINT_DIR = 'checkpoints' # 训练之后的模型储存的位置
    INCEPTION_MODEL_FILE = 'tensorflow_inception_graph.pb' # Inception-v3模型储存的位置

    # inception-v3模型参数
    BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'  # inception-v3模型中代表瓶颈层结果的张量名称
    JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'  # 图像输入张量对应的名称

    # 测试数据
    file_path = 'output\\out.jpg'
    #y_test = [0]

    # 读取数据
    image_data = tf.gfile.GFile(file_path, 'rb').read()

    # 评估
    checkpoint_file = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    with tf.Graph().as_default() as graph:
        with tf.Session().as_default() as sess:
            # 读取训练好的inception-v3模型
            with tf.gfile.GFile(INCEPTION_MODEL_FILE, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())

            # 加载inception-v3模型，并返回数据输入张量和瓶颈层输出张量
            bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(
                graph_def,
                return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME])

            # 使用inception-v3处理图片获取特征向量
            bottleneck_values = sess.run(bottleneck_tensor,
                                        {jpeg_data_tensor: image_data})
            # 将四维数组压缩成一维数组，由于全连接层输入时有batch的维度，所以用列表作为输入
            bottleneck_values = [np.squeeze(bottleneck_values)]

            # 加载元图和变量
            saver = tf.train.import_meta_graph('{}.meta'.format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # 通过名字从图中获取输入占位符
            input_x = graph.get_operation_by_name(
                'BottleneckInputPlaceholder').outputs[0]

            # 我们想要评估的tensors
            predictions = graph.get_operation_by_name('evaluation/ArgMax').outputs[
                0]

            # 收集预测值
            all_predictions = []
            all_predictions = sess.run(predictions, {input_x: bottleneck_values})

    # 如果提供了标签则打印正确率
    if True:
        #print(all_predictions) 打印预测标签
        print("\n检测结果：")
        if all_predictions == [0]:
            print("图片被裁剪过！\n")
        if all_predictions == [1]:
            print("图片添加过滤镜！\n")
        if all_predictions == [2]:
            print("图片被缩放过！\n")
        if all_predictions == [3]:
            print("图片被涂抹过！\n")
        if all_predictions == [4]:
            print("图片正常\n")


def IsValidImage(img_path):
    """
    判断文件是否为有效（完整）的图片
    :param img_path:图片路径
    :return:True：有效 False：无效
    """
    bValid = True
    try:
        Image.open(img_path).verify()
    except:
        bValid = False
    return bValid


def transimg(img_path):
    """
    转换图片格式
    :param img_path:图片路径
    :return: True：成功 False：失败
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
    print("启动中，请稍后...")
    eng = matlab.engine.start_matlab()
    while True:
        filepath1 = input("请输入需检测的图像路径: ")
        filepath2 = "output\\out.png"
        eng.extract(filepath1, filepath2, nargout = 0)
        filepath3 = "output\\out.png"
        transimg(filepath3)
        print("\n水印提取结束！开始检测，请稍后...")
        eval()
        print("\n选择您的下一步行动：\n    1.继续检测\n    2.退出")
        flag = input("请输入：")
        if (flag == '2'):
            eng.quit()
            exit()
        elif (flag != '1'):
            print("错误参数！将自动退出...")
            eng.quit()
            exit()
        os.system('cls')


if __name__ == "__main__":
    main()