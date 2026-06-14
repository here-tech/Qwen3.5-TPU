#!/usr/bin/env python3
"""HTTP API server for Qwen3.5 on BM1684x TPU — with Web chat UI."""

import argparse
import os
import sys
import time
import threading
import uuid

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import Qwen3_5


# --- Globals ---

model = None
model_lock = threading.Lock()
_model_path: str = ""
_config_path: str = ""
UPLOAD_DIR = ""

# --- Concurrency tracking ---
active_count = 0
active_lock = threading.Lock()
total_count = 0
total_lock = threading.Lock()
REQUEST_TIMEOUT = 300

# --- FastAPI app ---

app = FastAPI(title="aⁿ 乘方大数据 Qwen3.5-4B on Andata TPU", version="2.0")


@app.on_event("startup")
def startup():
    global model, UPLOAD_DIR
    UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    class Args:
        devid = 0
        model_path = _model_path
        config_path = _config_path
        video_ratio = 0.25

    model = Qwen3_5(Args())
    print(f"Model loaded. SEQLEN={model.model.SEQLEN} max_input={model.model.MAX_INPUT_LENGTH}")


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(503, "Model not loaded yet")
    busy = active_count > 0
    return {
        "status": "busy" if busy else "ok",
        "model": os.path.basename(_model_path),
        "active_requests": active_count,
        "total_requests": total_count,
    }


@app.post("/api/chat")
def chat(
    prompt: str = Form(...),
    file: UploadFile = File(None),
):
    if model is None:
        raise HTTPException(503, "Model not loaded yet")
    if not prompt.strip():
        raise HTTPException(400, "prompt is required")

    # Increment total request counter
    global total_count
    with total_lock:
        total_count += 1

    # Save uploaded file to temp location
    media_path = ""
    if file and file.filename:
        ext = os.path.splitext(file.filename)[1] or ".jpg"
        fname = f"{uuid.uuid4().hex}{ext}"
        media_path = os.path.join(UPLOAD_DIR, fname)
        with open(media_path, "wb") as f:
            f.write(file.file.read())

    # Try to acquire the model lock with timeout
    acquired = model_lock.acquire(timeout=REQUEST_TIMEOUT)
    if not acquired:
        # Clean up uploaded file
        if media_path and os.path.exists(media_path):
            try:
                os.remove(media_path)
            except OSError:
                pass
        raise HTTPException(503, "Server busy, request timed out waiting for model")

    global active_count
    try:
        with active_lock:
            active_count += 1
        text, ftl, tps = _run_inference(prompt, media_path)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {e}")
    finally:
        with active_lock:
            active_count -= 1
        model_lock.release()
        # Clean up uploaded file
        if media_path and os.path.exists(media_path):
            try:
                os.remove(media_path)
            except OSError:
                pass

    return {"text": text, "ftl": ftl, "tps": tps, "media_used": bool(media_path)}


@app.post("/api/clear")
def clear_history():
    """Clear the model's conversation history for a fresh session."""
    global model
    if model is not None:
        with model_lock:
            model.model.clear_history()
            model.history_max_posid = 0
    return {"status": "ok", "message": "History cleared"}


@app.get("/api/status")
def api_status():
    """Detailed server status with concurrency info."""
    if model is None:
        raise HTTPException(503, "Model not loaded yet")
    return {
        "status": "busy" if active_count > 0 else "idle",
        "model": os.path.basename(_model_path),
        "active_requests": active_count,
        "total_requests": total_count,
        "max_input_length": model.model.MAX_INPUT_LENGTH,
        "seqlength": model.model.SEQLEN,
    }


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE


def _run_inference(prompt: str, media_path: str):
    """Returns (text, ftl, tps)."""
    model.input_str = prompt
    media_path = (media_path or "").strip()

    if media_path == "":
        messages = model.text_message()
        media_type = "text"
    elif not os.path.exists(media_path):
        raise ValueError(f"Media file not found: {media_path}")
    else:
        media_type = model.get_media_type(media_path)
        if media_type == "image":
            messages = model.image_message(media_path)
        elif media_type == "video":
            messages = model.video_message(media_path)
        else:
            raise ValueError(f"Unsupported media type: {media_path}")

    inputs = model.process(messages, media_type)
    token_len = inputs.input_ids.numel()
    if token_len > model.model.MAX_INPUT_LENGTH:
        raise ValueError(
            f"Input too long: {token_len} tokens (max {model.model.MAX_INPUT_LENGTH})"
        )

    if model.support_history:
        if (token_len + model.model.history_length > model.model.SEQLEN - 128) or \
           (model.model.history_length > model.model.PREFILL_KV_LENGTH):
            model.model.clear_history()
            model.history_max_posid = 0

    first_start = time.time()
    model.model.forward_embed(inputs.input_ids.numpy())

    if media_type == "image":
        model.vit_process_image(inputs)
        position_ids = model.get_rope_index(
            inputs.input_ids, inputs.image_grid_thw, model.ID_IMAGE_PAD
        )
        model.max_posid = int(position_ids.max())
    elif media_type == "video":
        model.vit_process_video(inputs)
        position_ids = model.get_rope_index(
            inputs.input_ids, inputs.video_grid_thw, model.ID_VIDEO_PAD
        )
        model.max_posid = int(position_ids.max())
    else:
        position_ids = 3 * [list(range(token_len))]
        model.max_posid = token_len - 1

    token = model.forward_prefill(np.array(position_ids, dtype=np.int32))
    first_end = time.time()

    tok_num = 0
    full_word_tokens = []
    text = ""
    while token not in [model.ID_IM_END] and model.model.history_length < model.model.SEQLEN:
        full_word_tokens.append(token)
        word = model.tokenizer.decode(full_word_tokens, skip_special_tokens=True)
        if "�" not in word:
            if len(full_word_tokens) == 1:
                pre_word = word
                word = model.tokenizer.decode(
                    [token, token], skip_special_tokens=True
                )[len(pre_word):]
            text += word
            full_word_tokens = []
        model.max_posid += 1
        token = model.model.forward_next(
            np.array([model.max_posid, model.max_posid, model.max_posid], dtype=np.int32)
        )
        tok_num += 1

    model.history_max_posid = model.max_posid + 2
    next_end = time.time()

    ftl = first_end - first_start
    tps = tok_num / (next_end - first_end) if (next_end - first_end) > 0 else 0.0

    return text, round(ftl, 3), round(tps, 3)


# --- Web UI ---

HTML_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>aⁿ 乘方大数据 Qwen3.5-4B on Andata TPU</title>
<style>
html,body{height:100%}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  background:#0f0f0f;color:#e0e0e0;min-height:100dvh;display:flex;flex-direction:column}
.header{background:#1a1a1a;padding:12px 24px;border-bottom:1px solid #333;
  display:flex;align-items:center;justify-content:space-between;flex-shrink:0}
.header h1{font-size:18px;color:#fff;display:flex;align-items:center;gap:8px}
.status{font-size:12px;color:#4a4}
.content{flex:1;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:16px}
.message{max-width:85%;padding:12px 16px;border-radius:14px;line-height:1.6;white-space:pre-wrap;word-break:break-word}
.message.user{align-self:flex-end;background:#2563eb;color:#fff;border-bottom-right-radius:4px}
.message.assistant{align-self:flex-start;background:#1e1e1e;border:1px solid #333;border-bottom-left-radius:4px}
.message img{max-width:300px;max-height:300px;border-radius:8px;margin-bottom:8px;display:block}
.meta{font-size:11px;color:#888;margin-top:4px}
.footer{background:#1a1a1a;border-top:1px solid #333;padding:16px 24px;flex-shrink:0}
.input-row{display:flex;gap:10px;align-items:flex-end}
.input-row textarea{flex:1;background:#222;border:1px solid #444;border-radius:10px;color:#e0e0e0;
  padding:10px 14px;font-size:14px;resize:none;min-height:44px;max-height:30vh;outline:none;
  font-family:inherit}
.input-row textarea:focus{border-color:#2563eb}
.btn{padding:10px 16px;border:none;border-radius:10px;cursor:pointer;font-size:14px;font-weight:500;
  transition:background .2s;white-space:nowrap}
.btn-send{background:#2563eb;color:#fff}.btn-send:hover{background:#1d4ed8}
.btn-send:disabled{opacity:.5;cursor:not-allowed}
.btn-upload{background:#333;color:#ccc;position:relative;overflow:hidden}
.btn-upload:hover{background:#444}
.btn-upload input[type=file]{position:absolute;inset:0;opacity:0;cursor:pointer}
.btn-clear{background:transparent;color:#888;border:1px solid #444}
.btn-clear:hover{background:#333}
.preview-row{display:flex;gap:8px;margin-top:8px;flex-wrap:wrap}
.preview-item{position:relative;width:80px;height:80px;border-radius:8px;overflow:hidden;border:2px solid #444}
.preview-item img{width:100%;height:100%;object-fit:cover}
.preview-item .rm{position:absolute;top:2px;right:2px;background:#e53e3e;color:#fff;
  border:none;border-radius:50%;width:20px;height:20px;font-size:12px;cursor:pointer;line-height:1}
.loading{display:flex;align-items:center;gap:8px;color:#888;font-size:13px;padding:8px 0}
.loading .dot{width:6px;height:6px;background:#2563eb;border-radius:50%;animation:bounce 1.4s infinite}
.loading .dot:nth-child(2){animation-delay:.2s}
.loading .dot:nth-child(3){animation-delay:.4s}
@keyframes bounce{0%,80%,100%{transform:scale(1)}40%{transform:scale(1.5)}}
@media (max-width:600px){
  .header{padding:10px 14px}
  .header h1{font-size:15px}
  .content{padding:12px}
  .footer{padding:10px 14px}
  .input-row{flex-wrap:wrap;gap:6px}
  .btn{padding:8px 12px;font-size:12px}
  .message{max-width:95%}
}
</style>
</head>
<body>
<div class="header">
  <h1>aⁿ 乘方大数据 Qwen3.5-4B <span style="font-weight:400;font-size:13px;color:#888">on Andata TPU</span></h1>
  <span class="status" id="status">&#x25cf; 就绪</span>
</div>
<div class="content" id="chat"></div>
<div class="footer">
  <div class="preview-row" id="preview"></div>
  <div class="input-row">
    <button class="btn btn-upload">&#x1f4c1; 上传图片<input type="file" id="fileInput" accept="image/*"></button>
    <textarea id="prompt" rows="1" placeholder="输入消息... (Enter 发送, Shift+Enter 换行)"></textarea>
    <button class="btn btn-send" id="sendBtn" onclick="send()">&#x27a4; 发送</button>
    <button class="btn btn-clear" onclick="clearChat()">清空</button>
  </div>
</div>

<script>
let selectedFile = null;
const chat = document.getElementById('chat');
const prompt = document.getElementById('prompt');
const status = document.getElementById('status');
const preview = document.getElementById('preview');

prompt.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});


prompt.addEventListener('input', () => {
  prompt.style.height = 'auto'; 
  prompt.style.height = prompt.scrollHeight + 'px';
});
document.getElementById('fileInput').addEventListener('change', e => {
  const f = e.target.files[0];
  if (!f) return;
  selectedFile = f;
  const reader = new FileReader();
  reader.onload = ev => {
    preview.innerHTML = `<div class="preview-item">
      <img src="${ev.target.result}"><button class="rm" onclick="removeFile()">x</button></div>`;
  };
  reader.readAsDataURL(f);
});

function removeFile() {
  selectedFile = null;
  preview.innerHTML = '';
  document.getElementById('fileInput').value = '';
}

function clearChat() {
  chat.innerHTML = '';
  fetch('/api/clear', {method:'POST'}).catch(()=>{});
}

function addMessage(role, text, meta) {
  const div = document.createElement('div');
  div.className = 'message ' + role;
  div.textContent = text;
  if (meta) div.innerHTML += '<div class="meta">' + meta + '</div>';
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

function addImage(src) {
  const div = document.createElement('div');
  div.className = 'message user';
  div.innerHTML = '<img src="' + src + '">';
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

function setLoading(show) {
  document.getElementById('sendBtn').disabled = show;
  if (show) {
    status.innerHTML = '&#x25cf; <span class="loading">推理中<span class="dot"></span><span class="dot"></span><span class="dot"></span></span>';
    status.style.color = '#ea0';
  } else {
    status.innerHTML = '&#x25cf; 就绪';
    status.style.color = '#4a4';
  }
}

async function send() {
  const text = prompt.value.trim();
  if (!text) return;
  setLoading(true);
  prompt.value = '';
  prompt.style.height = 'auto';

  // Show user message
  if (selectedFile) {
    addImage(URL.createObjectURL(selectedFile));
  }
  addMessage('user', text);

  const form = new FormData();
  form.append('prompt', text);
  if (selectedFile) form.append('file', selectedFile);
  removeFile();

  try {
    const res = await fetch('/api/chat', { method: 'POST', body: form });
    const data = await res.json();
    if (data.text) {
      addMessage('assistant', data.text,
        'FTL: ' + data.ftl + 's | TPS: ' + data.tps + ' tok/s' +
        (data.media_used ? ' | &#x1f5bc;' : ''));
    } else {
      addMessage('assistant', 'Error: ' + JSON.stringify(data));
    }
  } catch(e) {
    addMessage('assistant', 'Network error: ' + e.message);
  }
  setLoading(false);
}
</script>
</body>
</html>"""


# --- Entrypoint ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3.5 HTTP Server")
    parser.add_argument("-m", "--model_path", type=str, required=True,
                        help="Path to bmodel file")
    parser.add_argument("-c", "--config_path", type=str, default="../config",
                        help="Path to processor config")
    parser.add_argument("--port", type=int, default=8080, help="Listen port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Listen host")
    args = parser.parse_args()

    _model_path = os.path.abspath(args.model_path)
    _config_path = os.path.abspath(args.config_path)

    uvicorn.run(app, host=args.host, port=args.port)
