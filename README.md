# Qwen3.5-4B on Andata TPU

基于 SOPHGO BM1684x TPU 的 Qwen3.5-4B 多模态大模型推理部署方案，由 **aⁿ 乘方大数据** 提供。

## 项目简介

本项目将 Qwen3.5-4B 模型量化部署到比特大陆 BM1684x TPU 计算卡上，提供高性能、低功耗的推理服务。支持：

- **文本对话** — 流式文本生成，支持多轮对话上下文
- **图像理解** — 支持 JPG/PNG/GIF/BMP/WebP 图片输入
- **视频理解** — 支持 MP4/AVI/MOV/MKV 等视频输入
- **Web Chat UI** — 内置美观的网页聊天界面
- **HTTP API** — 标准 RESTful API，方便集成

### 模型架构

| 参数 | 值 |
|------|-----|
| 模型 | Qwen3.5-4B (Qwen3_5ForConditionalGeneration) |
| 量化 | INT4 AutoRound W4BF16 |
| 隐藏层维度 | 1024 |
| Transformer 层数 | 24 (6层 full-attention + 18层 linear-attention) |
| 注意力头数 | 8 |
| 词表大小 | 248,320 |
| 视觉编码器 | 12层 ViT, hidden_size=768 |
| 最大序列长度 | 2048 tokens |
| 最大输入长度 | 1024 tokens |

## 硬件要求

- **TPU**: SOPHGO BM1684x 计算卡 (8GB 显存)
- **功耗**: ~25W (TPU only)
- **系统**: Linux (aarch64), Debian/Ubuntu
- **内存**: ≥ 4GB RAM
- **存储**: ≥ 10GB 可用空间

当前已占用 TPU 显存: 3852MB / 8192MB

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

下载 bmodel 文件到本地：

```bash
# 从 GitHub LFS 下载 (链接见 Release 页面)
# 或联系 aⁿ 乘方大数据获取模型文件

# 放置到指定位置
mkdir -p /data/qwen
cp qwen3.5-4b-int4-autoround_w4bf16_seq2048_bm1684x_1dev_dynamic_20260416_144422.bmodel /data/qwen/
```

### 4. 编译 C++ 扩展 (如需要)

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

python server.py \
  -m /data/qwen/qwen3.5-4b-int4-autoround_w4bf16_seq2048_bm1684x_1dev_dynamic_20260416_144422.bmodel \
  -c ./config \
  --port 8080 \
  --host 0.0.0.0
```

启动后访问 `http://<服务器IP>:8080` 即可使用 Web Chat UI。

### 命令行模式

```bash
cd Qwen3.5
conda activate qwen3.5

# 单次推理
python pipeline.py \
  -m /data/qwen/qwen3.5-4b-int4-autoround_w4bf16_seq2048_bm1684x_1dev_dynamic_20260416_144422.bmodel \
  -c ./config \
  -p "你好，请介绍一下你自己"

# 交互式对话
python pipeline.py \
  -m /data/qwen/qwen3.5-4b-int4-autoround_w4bf16_seq2048_bm1684x_1dev_dynamic_20260416_144422.bmodel \
  -c ./config
```

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
  "model": "qwen3.5-4b-int4-autoround_w4bf16_seq2048_bm1684x_1dev_dynamic_20260416_144422.bmodel",
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
  "model": "qwen3.5-4b-int4-autoround_w4bf16_seq2048_bm1684x_1dev_dynamic_20260416_144422.bmodel",
  "active_requests": 0,
  "total_requests": 10,
  "max_input_length": 1024,
  "seqlength": 2048
}
```

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

在 BM1684x TPU 上的实测性能：

- **FTL (首Token延迟)**: ~0.2s
- **TPS (生成速度)**: ~15 tokens/s
- **TPU 显存占用**: 3852MB / 8192MB
- **TPU 功耗**: ~25W

## 致谢

- [SOPHGO](https://github.com/sophgo) — TPU-MLIR 工具链和 BM1684x 硬件平台
- [Qwen](https://github.com/QwenLM/Qwen) — Qwen3.5 基础模型
- **aⁿ 乘方大数据** — 模型量化和部署优化

## License

本项目基于 Apache 2.0 许可证。模型权重遵循 Qwen 原始许可证。
