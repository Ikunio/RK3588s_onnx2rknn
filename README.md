RK3588s_onnx2rknn
基于瑞微芯RKNN工具链,本仓库只是用于学习和记录环境配置
板端配置：orangepi5Pro，ubuntu22.04下进行
主机配置：x64，Vmware，ubuntu20.04


一、配置Ubuntu PC端转模型
1.在 Ubuntu PC 端安装 RKNN-Toolkit2
Toolkit2 是一款在 Ubuntu PC 平台上使用的开发套件,用户使用该工具提供的Python 接口可以便捷地完成模型转换、推理和性能评估等功能

2.在 Ubuntu PC 端，打开一个命令行窗口，然后输入以下命令来安装 python3 和 pip3
test@test:~$ sudo apt-get install python3 python3-dev python3-pip

3.可以使用以下命令来查看已安装的 python3 的版本
test@test:~$ python3 --version
Python 3.8.10

4.然后输入以下命令来下载 1.5.2 版本的 RKNN-Toolkit2
test@test:~$ git clone git clone https://github.com/airockchip/rknn-toolkit2 -b v1.5.2

5.然后输入以下命令安装 python3 相应版本的依赖包，这个命令将使用 pip3 安装文件 requirements_cp38-1.5.2.txt 中列出的依赖项。如果依赖安装不全的话就不指定安
装源单独安装里面的每个包。
test@test:~$ pip3 install -r rknn-toolkit2/doc/requirements_cp38-1.5.2.txt -i \
https://mirror.baidu.com/pypi/simple

6.然后输入以下命令使用 pip3 来安装 RKNN-Toolkit2 软件包，安装完成后就可以使用 RKNN-Toolkit2 了
test@test:~$ pip3 install rknn-toolkit2/packages/rknn_toolkit2-1.5.2+b642f30c-cp38-cp38-linux_x86_64.whl

7.使用 RKNN-Toolkit2 进行模型转换和模型推理RKNN-Toolkit2 支持将 Caffe、TensorFlow、TensorFlow Lite、ONNX、DarkNet、 PyTorch 等模型转为 RKNN 模型，然后在 Ubuntu PC 端通过仿真或使用开发板的 NPU 运行 RKNN 模型来进行推理。
现在使用yolov5的转换示例

8.在 Ubuntu PC 端仿真运行模型RKNN-Toolkit2 搭载了内置模拟器，可以让用户在 Ubuntu PC 端模拟模型在Rockchip NPU 上的推理过程。这样模型转换和推理均可以在 Ubuntu PC 端完成，从而帮助用户更快地测试和验证他们的模型。

9.首先切换到 rknn-toolkit2/examples/onnx/yolov5 目录
test@test:~$ cd rknn-toolkit2/examples/onnx/yolov5/

10.然后运行 test.py 脚本，该脚本首先将 yolov5s_relu.onnx 模型转换为可以在模拟器上运行的 RKNN 模型，然后使用模拟器仿真运行该模型对当前目录下的 bus.jpg 图
片进行推理
test@test:~/rknn-toolkit2/examples/onnx/yolov5$ python3 test.py

11.转换得到的模型文件 yolov5s_relu.rknn 和推理的图片结果 result.jpg 保存在当前目录下



二、在香橙派5Pro中的环境配置
1.在终端输入
git clone https://github.com/rockchip-linux/rknpu2 -b v1.5.2

2.在终端输入
sudo cp -r rknpu2/runtime/RK3588/Linux/rknn_server/aarch64/usr/bin/* /usr/bin/

3.在终端输入
sudo cp rknpu2/runtime/RK3588/Linux/librknn_api/aarch64/librknnrt.so /usr/lib/

4.打开开发板的rknn_server服务
sudo restart_rknn.sh

5.验证环境是否搭好
pgrep rknn_server
