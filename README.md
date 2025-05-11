# RK3588S ONNX 转 RKNN 模型工具链配置指南
## 目前这个代码中只包含了PC端的模型转换
## 而rknn的板端推理可以去我主页里的rknn_ros2一起使用
🚀 基于瑞芯微 RKNN 工具链的模型转换环境配置文档  
📌 适用场景：学习记录 | 模型转换 | NPU 环境部署  

---

## 🖥️ 硬件环境

| 设备类型   | 配置说明                                 |
|------------|------------------------------------------|
| **开发板** | OrangePi 5 Pro (RK3588S, Ubuntu 22.04)   |
| **主机**   | x64 架构 + VMware + Ubuntu 20.04         |

---

## 📋 目录

1. [PC 端模型转换环境配置](#-pc-端模型转换环境配置)  
2. [开发板推理环境配置](#-开发板环境配置)  

---

## 🔧 PC 端模型转换环境配置

### 1. 安装 Python 基础环境

```bash
sudo apt-get update
sudo apt-get install python3 python3-dev python3-pip
python3 --version  # 推荐版本: Python 3.8.x
```

### 2. 获取 RKNN-Toolkit2（版本 1.5.2）

```
git clone https://github.com/airockchip/rknn-toolkit2 -b v1.5.2
```

### 3. 安装依赖项

```
pip3 install -r rknn-toolkit2/doc/requirements_cp38-1.5.2.txt -i https://mirror.baidu.com/pypi/simple
```

> 💡 若某些依赖无法安装，可手动逐个安装 requirements 文件中列出的包。

### 4. 安装 RKNN-Toolkit2 主包

```
pip3 install rknn-toolkit2/packages/rknn_toolkit2-1.5.2+b642f30c-cp38-cp38-linux_x86_64.whl
```

---

## 🧪 ONNX 模型转换示例（YOLOv5）

### 1. 进入示例目录

```
cd rknn-toolkit2/examples/onnx/yolov5/
```

### 2. 执行模型转换与推理

```
python3 test.py
```

该脚本将执行以下操作：

- 将 \`yolov5s_relu.onnx\` 转换为 \`yolov5s_relu.rknn\`
- 使用模拟器对 \`bus.jpg\` 图片进行推理
- 结果图像输出为 \`result.jpg\`

---

## 🛠️ 开发板环境配置（OrangePi 5 Pro）

### 1. 获取运行时库 rknpu2

```
git clone https://github.com/rockchip-linux/rknpu2 -b v1.5.2
```

### 2. 拷贝运行依赖文件至系统路径

```
sudo cp -r rknpu2/runtime/RK3588/Linux/rknn_server/aarch64/usr/bin/* /usr/bin/
sudo cp rknpu2/runtime/RK3588/Linux/librknn_api/aarch64/librknnrt.so /usr/lib/
```

### 3. 启动 rknn_server 服务

```
sudo restart_rknn.sh
```

---

## ✅ 验证与测试

### 检查 rknn_server 是否成功运行

```
pgrep rknn_server
```

若成功启动，将返回对应进程 PID。

---

🔚 至此，RKNN 工具链模型转换和板端部署环境配置完成，可用于后续深度学习模型部署与测试。欢迎Start
