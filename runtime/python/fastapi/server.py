# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse
import tempfile
import io
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from fastapi import FastAPI, UploadFile, Form, File, BackgroundTasks
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch
import torchaudio
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/../../..'.format(ROOT_DIR))
sys.path.append('{}/../../../third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import AutoModel

app = FastAPI()
# set cross region allowance
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])


def _collect_and_pack_wav(model_output) -> bytes:
    """收集模型所有输出 chunk，拼接后打包成完整 WAV（与 example.py 的 torchaudio.save 一致）。"""
    chunks = []
    for chunk in model_output:
        chunks.append(chunk['tts_speech'])
    if not chunks:
        return b''
    speech = torch.cat(chunks, dim=1)
    buf = io.BytesIO()
    torchaudio.save(buf, speech, cosyvoice.sample_rate, format='wav')
    return buf.getvalue()


def _save_upload_to_tmp(upload_file: UploadFile) -> str:
    """把上传文件存为临时 wav，返回路径。调用方负责用 BackgroundTasks 清理。"""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.write(upload_file.file.read())
    tmp.close()
    return tmp.name


@app.get("/inference_sft")
@app.post("/inference_sft")
async def inference_sft(tts_text: str = Form(), spk_id: str = Form()):
    model_output = cosyvoice.inference_sft(tts_text, spk_id, stream=False)
    wav_bytes = _collect_and_pack_wav(model_output)
    return Response(content=wav_bytes, media_type="audio/wav")


@app.get("/inference_zero_shot")
@app.post("/inference_zero_shot")
async def inference_zero_shot(
    tts_text: str = Form(),
    prompt_text: str = Form(),
    prompt_wav: UploadFile = File(),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    tmp_path = _save_upload_to_tmp(prompt_wav)
    background_tasks.add_task(os.unlink, tmp_path)
    model_output = cosyvoice.inference_zero_shot(tts_text, prompt_text, tmp_path, stream=False)
    wav_bytes = _collect_and_pack_wav(model_output)
    return Response(content=wav_bytes, media_type="audio/wav")


@app.get("/inference_cross_lingual")
@app.post("/inference_cross_lingual")
async def inference_cross_lingual(
    tts_text: str = Form(),
    prompt_wav: UploadFile = File(),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    tmp_path = _save_upload_to_tmp(prompt_wav)
    background_tasks.add_task(os.unlink, tmp_path)
    model_output = cosyvoice.inference_cross_lingual(tts_text, tmp_path, stream=False)
    wav_bytes = _collect_and_pack_wav(model_output)
    return Response(content=wav_bytes, media_type="audio/wav")


# @app.get("/inference_instruct")
# @app.post("/inference_instruct")
# async def inference_instruct(tts_text: str = Form(), spk_id: str = Form(), instruct_text: str = Form()):
#     model_output = cosyvoice.inference_instruct(tts_text, spk_id, instruct_text, stream=False)
#     wav_bytes = _collect_and_pack_wav(model_output)
#     return Response(content=wav_bytes, media_type="audio/wav")


@app.get("/inference_instruct2")
@app.post("/inference_instruct2")
async def inference_instruct2(
    tts_text: str = Form(),
    instruct_text: str = Form(),
    prompt_wav: UploadFile = File(),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    tmp_path = _save_upload_to_tmp(prompt_wav)
    background_tasks.add_task(os.unlink, tmp_path)
    model_output = cosyvoice.inference_instruct2(tts_text, instruct_text, tmp_path, stream=False)
    wav_bytes = _collect_and_pack_wav(model_output)
    return Response(content=wav_bytes, media_type="audio/wav")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=50000)
    parser.add_argument('--model_dir',
                        type=str,
                        default='FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
                        help='local path or modelscope repo id')
    args = parser.parse_args()
    cosyvoice = AutoModel(model_dir=args.model_dir)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
