# Qwen3.5-4B on Andata TPU webservice

由 **aⁿ 乘方大数据** 提供基于 TPU 的 Qwen3.5-4B 多模态大模型推理 Web 服务。

## 项目简介

本项目将 Qwen3.5-4B 模型量化部署到比特大陆 BM1684x TPU 计算卡上，提供 HTTP API + Web Chat UI 的完整 Web 服务。同时支持 Qwen3.5-9B 模型，只需替换 bmodel 文件并修改启动参数即可切换。功能包括：

- **文本对话** — 流式文本生成，支持多轮对话上下文
- **图像理解** — 支持 JPG/PNG/GIF/BMP/WebP 图片输入
- **视频理解** — 支持 MP4/AVI/MOV/MKV 等视频输入
- **Web Chat UI** — 内置美观的网页聊天界面
- **HTTP API** — 标准 RESTful API，方便集成

### 模型架构

| 参数 | Qwen3.5-4B | Qwen3.5-9B |
|------|-----------|-----------|
| 模型架构 | Qwen3_5ForConditionalGeneration | Qwen3_5ForConditionalGeneration |
| 量化 | INT4 AutoRound W4BF16 | INT4 AutoRound W4BF16 |
| 隐藏层维度 | 1024 | — |
| Transformer 层数 | 24 (6 full + 18 linear) | — |
| 注意力头数 | 8 | — |
| 词表大小 | 248,320 | — |
| 视觉编码器 | 12层 ViT, hidden=768 | — |
| 最大序列长度 | 2048 tokens | 2048 tokens |
| 最大输入长度 | 1024 tokens | — |

## 并发处理机制

服务支持多用户并发访问，核心设计：

- **模型单例** — 一个 TPU 设备运行一个模型实例，所有用户共享
- **锁超时机制** — `threading.Lock` 保护模型推理，超时 300 秒后返回 503
- **请求追踪** — `active_count` / `total_count` 实时监控并发状态
- **健康感知** — `/health` 和 `/api/status` 区分 `idle`/`busy`/`ok` 状态

```
用户A ──┐                ┌── TPU Model ──► 响应A
        ├── Lock Queue ──┤
用户B ──┘   (timeout=300s) └──► 响应B
```

## 性能指标

在 BM1684x TPU 上的实测性能 (Qwen3.5-4B)：

- **FTL (首Token延迟)**: ~0.2s
- **TPS (生成速度)**: ~15 tokens/s
- **TPU 显存占用**: 3852MB / 8192MB
- **TPU 功耗**: ~20-40W

## 硬件要求

- **TPU**: SOPHGO BM1684x 计算卡 (8GB 显存)
- **功耗**: ~25W (TPU only)
- **系统**: Linux (aarch64), Debian/Ubuntu
- **内存**: ≥ 4GB RAM
- **存储**: ≥ 10GB 可用空间
- Qwen3.5-4B TPU 显存占用: 3852MB / 8192MB
  
<img width="1800" height="1434" alt="Screenshot_2026-06-14-17-18-18-570_com microsoft emmx-edit" src="https://github.com/user-attachments/assets/3f6b5943-c239-4083-b989-2db96bd45795" />



## 项目结构

```
Qwen3.5/
├── README.md              # 本文件
├── server.py              # HTTP API 服务器 + Web Chat UI
├── pipeline.py            # 模型推理流水线
├── chat.cpp               # C++ TPU 推理绑定 (BM1684x SDK)
├── chat.cpython-310-aarch64-linux-gnu.so  # 编译后的 Python 扩展
├── CMakeLists.txt         # C++ 扩展构建配置
├── config/                # 模型配置文件 (tokenizer, processor)
│   ├── config.json        # 模型架构配置
│   ├── tokenizer.json     # Tokenizer
│   ├── vocab.json         # 词表
│   ├── merges.txt         # BPE merges
│   ├── tokenizer_config.json
│   ├── chat_template.jinja
│   ├── preprocessor_config.json
│   └── video_preprocessor_config.json
└── uploads/               # 上传文件临时目录 (自动创建)
```

## 安装方法

### 1. 安装 TPU 驱动和 SDK

确保 BM1684x 驱动已正确安装：

```bash
# 检查 TPU 设备
ls /dev/bm-tpu0

# 安装 SOPHGO SDK (如未安装)
# 参考: https://github.com/sophgo/sophon-demo
```

### 2. 安装 Conda 环境

```bash
# 安装 Miniconda (如未安装)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
bash Miniconda3-latest-Linux-aarch64.sh

# 创建并激活环境
conda create -n qwen3.5 python=3.10 -y
conda activate qwen3.5

# 安装依赖
pip install numpy fastapi uvicorn python-multipart pydantic
pip install transformers qwen-vl-utils torch
```

### 3. 下载模型文件

使用 SOPHGO 官方 dfss 工具下载 bmodel 文件。

**Qwen3.5-4B 模型：**

```bash
# 安装 dfss
pip install dfss

# 下载 4B 模型
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3.5-4b-int4-autoround_w4bf16_seq2048_bm1684x_1dev_dynamic_20260416_144422.bmodel

# 将下载的 bmodel 文件移动到项目目录 (或任意路径)
mv qwen3.5-4b-int4-autoround_w4bf16_seq2048_bm1684x_1dev_dynamic_20260416_144422.bmodel ./Qwen3.5/
```

**Qwen3.5-9B 模型：**

```bash
# 下载 9B 模型
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3.5-9b-int4-autoround_w4bf16_seq2048_bm1684x_1dev_dynamic_20260416_150658.bmodel

# 移动到项目目录
mv qwen3.5-9b-int4-autoround_w4bf16_seq2048_bm1684x_1dev_dynamic_20260416_150658.bmodel ./Qwen3.5/
```

### 4. 修改 Python 文件中的模型路径

下载 bmodel 文件后，在启动服务时需要指定正确的 bmodel 路径。有两种方式：

**方式一：启动时通过命令行参数指定 (推荐)**

启动 server.py 时通过 `-m` 参数指定 bmodel 文件路径，无需修改代码：

```bash
# 4B 模型
python server.py -m ./qwen3.5-4b-int4-autoround_w4bf16_seq2048_bm1684x_1dev_dynamic_20260416_144422.bmodel -c ./config --port 8080

# 9B 模型 (替换为 9B 的 bmodel 路径即可)
python server.py -m ./qwen3.5-9b-int4-autoround_w4bf16_seq2048_bm1684x_1dev_dynamic_20260416_150658.bmodel -c ./config --port 8080
```

**方式二：修改 server.py 中的默认路径**

如果希望固定模型路径，可以直接修改 `server.py` 中 `__main__` 段的默认值。找到文件末尾的这段代码并修改 `model_path` 默认值：

```python
# server.py 末尾 (约第 200 行附近)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3.5 HTTP Server")
    parser.add_argument("-m", "--model_path", type=str, required=True,   # <-- 将 required=True 改为 default="./your_model.bmodel"
                        help="Path to bmodel file")
    parser.add_argument("-c", "--config_path", type=str, default="../config",
                        help="Path to processor config")
    ...
```

同样，`pipeline.py` 中的模型路径也是通过 `-m` / `--model_path` 参数传入，切换模型时只需修改启动命令中的路径。

### 5. 编译 C++ 扩展 (如需要)

如果系统上没有预编译的 `.so` 文件，需要重新编译：

```bash
cd Qwen3.5
mkdir -p build && cd build
cmake ..
make
cp chat.cpython-310-aarch64-linux-gnu.so ../
```

## 启动服务

### HTTP Server 模式 (推荐)

```bash
cd Qwen3.5
conda activate qwen3.5

# 启动 4B 模型
python server.py \
  -m ./qwen3.5-4b-int4-autoround_w4bf16_seq2048_bm1684x_1dev_dynamic_20260416_144422.bmodel \
  -c ./config \
  --port 8080 \
  --host 0.0.0.0

# 启动 9B 模型 (仅更换 bmodel 文件路径)
python server.py \
  -m ./qwen3.5-9b-int4-autoround_w4bf16_seq2048_bm1684x_1dev_dynamic_20260416_150658.bmodel \
  -c ./config \
  --port 8080 \
  --host 0.0.0.0
```

启动后访问 `http://<服务器IP>:8080` 即可使用 Web Chat UI。

### 命令行模式

```bash
cd Qwen3.5
conda activate qwen3.5

# 4B 单次推理
python pipeline.py \
  -m ./qwen3.5-4b-int4-autoround_w4bf16_seq2048_bm1684x_1dev_dynamic_20260416_144422.bmodel \
  -c ./config \
  -p "你好，请介绍一下你自己"

# 9B 单次推理
python pipeline.py \
  -m ./qwen3.5-9b-int4-autoround_w4bf16_seq2048_bm1684x_1dev_dynamic_20260416_150658.bmodel \
  -c ./config \
  -p "你好，请介绍一下你自己"

# 交互式对话
python pipeline.py \
  -m ./qwen3.5-4b-int4-autoround_w4bf16_seq2048_bm1684x_1dev_dynamic_20260416_144422.bmodel \
  -c ./config
```

> **切换模型提示**：切换 4B/9B 模型只需修改 `-m` 参数指定的 bmodel 文件路径，config 目录和其余参数无需更改。

## API 文档

### 接口列表

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | Web Chat UI 界面 |
| GET | `/health` | 健康检查 + 模型状态 |
| GET | `/api/status` | 详细状态 (并发数、序列长度等) |
| POST | `/api/chat` | 对话接口 (支持图片/视频) |
| POST | `/api/clear` | 清空对话历史 |

### 对话接口

```bash
# 纯文本对话
curl -X POST http://localhost:8080/api/chat \
  -F "prompt=你好，介绍一下BM1684x TPU"

# 图片理解
curl -X POST http://localhost:8080/api/chat \
  -F "prompt=这张图片里有什么？" \
  -F "file=@/path/to/image.jpg"
```

响应格式：

```json
{
  "text": "你好！有什么我可以帮你的吗？",
  "ftl": 0.203,
  "tps": 14.877,
  "media_used": false
}
```

| 字段 | 说明 |
|------|------|
| text | 模型生成的回复文本 |
| ftl | First Token Latency (首token延迟，秒) |
| tps | Tokens Per Second (生成速度) |
| media_used | 是否使用了图片/视频输入 |

### 健康检查

```bash
curl http://localhost:8080/health
```

```json
{
  "status": "ok",
  "model": "qwen3.5-4b-...",
  "active_requests": 0,
  "total_requests": 10
}
```

### 服务状态

```bash
curl http://localhost:8080/api/status
```

```json
{
  "status": "idle",
  "model": "qwen3.5-4b-...",
  "active_requests": 0,
  "total_requests": 10,
  "max_input_length": 1024,
  "seqlength": 2048
}
```

## 致谢

- [SOPHGO](https://github.com/sophgo) — TPU-MLIR 工具链和 BM1684x 硬件平台
- [Qwen](https://github.com/QwenLM/Qwen) — Qwen3.5 基础模型
- **aⁿ 乘方大数据** — 模型量化和部署优化

## License

本项目基于 Apache 2.0 许可证。模型权重遵循 Qwen 原始许可证。
