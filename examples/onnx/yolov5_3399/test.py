import os
import urllib
import traceback
import time
import sys
from rknn.api import RKNN

if __name__ == '__main__':
    #参数设置
    ONNX_MODEL = 'apple4_7.onnx'   # onnx模型路径
    RKNN_MODEL = '/home/ros/rknn/rknn-toolkit2/examples/onnx/yolov5_3399/best.rknn'     # rknn模型保存路径
    DATASET_IMAGES = 'images' # 量化数据集路径,会自动生成'dataset.txt'，无需自己建'dataset.txt'  
    IMG_SIZE = 640 #图片训练尺寸
    outputs=['326','378','430']#访问https://netron.app/，将onnx拖进查看最后三层卷积层输出信息进行确定
    #开始转换
    # 量化数据文本生成，自动生成'dataset.txt'
    image_files = os.listdir(DATASET_IMAGES)
    image_files=[x for x in image_files if '.ipynb' not in x]
    DATASET='dataset.txt' 
    with open(DATASET, 'w') as f: #自动生成data.txt
        for i in image_files:  # i 是目录下的文件或文件夹
            i_old = os.path.join(DATASET_IMAGES,i)  # 字符串拼接
            print(i_old)
            f.write(i_old + '\n')
        
    # 创建rknn实例
    rknn = RKNN()

    if not os.path.exists(ONNX_MODEL):
        print('model not exist')
        exit(-1)
    
    # 预处理设置
    print('--> Configuring model')
    QUANTIZE_ON = True
    rknn.config(batch_size=32,
                reorder_channel='0 1 2',
                mean_values=[[0, 0, 0]],
                std_values=[[255, 255, 255]],
                optimization_level=3,
                target_platform = 'rk3399pro',
                output_optimize=1,
                quantize_input_node=QUANTIZE_ON)
    print('Configuration is done')

    # 加载onnx模型
    print('--> Loading model')
    #访问https://netron.app/，将onnx拖进查看最后三层卷积层输出信息进行确定
    ret = rknn.load_onnx(model=ONNX_MODEL,outputs=outputs)
    
    # 加载失败则退出
    if ret != 0:
        print('Load yolov5 failed!')
        exit(ret)
    print('Loading is done')

    # 构建模型
    print('--> Building model')
    ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET)
    if ret != 0:
        print('Build yolov5 failed!')
        exit(ret)
    print('Building is done')

    # 导出模型
    print('--> Exporting RKNN model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export yolov5rknn failed!')
        exit(ret)
    print('Export is done')
