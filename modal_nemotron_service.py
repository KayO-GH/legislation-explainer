"""Modal deployment for an OpenAI-compatible Nemotron generator endpoint.

This service solves the current Hugging Face router gap for
`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` by serving the model directly on
Modal with SGLang. The app can then point `NEMOTRON_BASE_URL` at the deployed
Modal URL and continue using the existing OpenAI client path.
"""

from __future__ import annotations

import os
import subprocess
import time

import modal
import modal.experimental

APP_NAME = os.getenv("MODAL_APP_NAME", "legislation-explainer-nemotron")
MODEL_NAME = os.getenv(
    "NEMOTRON_MODAL_MODEL",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
)
PORT = int(os.getenv("NEMOTRON_MODAL_PORT", "8000"))
GPU = os.getenv("NEMOTRON_MODAL_GPU", "H100")
REGION = os.getenv("NEMOTRON_MODAL_REGION", "us-east")
PROXY_REGION = os.getenv("NEMOTRON_MODAL_PROXY_REGION", "us-east")
MIN_CONTAINERS = int(os.getenv("NEMOTRON_MODAL_MIN_CONTAINERS", "0"))
SCALEDOWN_WINDOW = int(os.getenv("NEMOTRON_MODAL_SCALEDOWN_WINDOW", "300"))
TIMEOUT_SECONDS = int(os.getenv("NEMOTRON_MODAL_TIMEOUT_SECONDS", "3600"))
STARTUP_TIMEOUT_SECONDS = int(os.getenv("NEMOTRON_MODAL_STARTUP_TIMEOUT_SECONDS", "3600"))
TARGET_INPUTS = int(os.getenv("NEMOTRON_MODAL_TARGET_INPUTS", "8"))
HF_SECRET_NAME = os.getenv("NEMOTRON_MODAL_HF_SECRET_NAME", "huggingface-secret")
VOLUME_NAME = os.getenv("NEMOTRON_MODAL_VOLUME_NAME", "legislation-explainer-nemotron-cache")
HF_CACHE_PATH = "/root/.cache/huggingface"

server_image = (
    modal.Image.from_registry("lmsysorg/sglang:v0.5.11")
    .entrypoint([])
    .run_commands("rm -rf /root/.cache/huggingface")
    .env(
        {
            "HF_HUB_CACHE": HF_CACHE_PATH,
            "HF_XET_HIGH_PERFORMANCE": "1",
            "SAFETENSORS_FAST_GPU": "1",
            "NVIDIA_TF32_OVERRIDE": "1",
            "SGLANG_ENABLE_SPEC_V2": "1",
            "SGLANG_ENABLE_JIT_DEEPGEMM": "0",
        }
    )
)

with server_image.imports():
    import requests

app = modal.App(APP_NAME, image=server_image)
hf_secret = modal.Secret.from_name(HF_SECRET_NAME)
hf_cache_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _check_running(process: subprocess.Popen[str]) -> None:
    return_code = process.poll()
    if return_code is not None:
        raise subprocess.CalledProcessError(return_code, cmd=process.args)


def _wait_ready(process: subprocess.Popen[str], timeout_seconds: int = 1800) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            _check_running(process)
            response = requests.get(f"http://127.0.0.1:{PORT}/health", timeout=5)
            response.raise_for_status()
            return
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.HTTPError,
            requests.exceptions.Timeout,
            subprocess.CalledProcessError,
        ):
            time.sleep(5)
    raise TimeoutError(f"SGLang server was not ready within {timeout_seconds} seconds.")


def _warm_up_server() -> None:
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "Reply with the single word READY."}],
        "max_tokens": 8,
        "temperature": 0,
    }
    for _ in range(2):
        response = requests.post(
            f"http://127.0.0.1:{PORT}/v1/chat/completions",
            json=payload,
            timeout=30,
        )
        response.raise_for_status()


@app.cls(
    gpu=GPU,
    min_containers=MIN_CONTAINERS,
    scaledown_window=SCALEDOWN_WINDOW,
    timeout=TIMEOUT_SECONDS,
    startup_timeout=STARTUP_TIMEOUT_SECONDS,
    region=REGION,
    secrets=[hf_secret],
    volumes={HF_CACHE_PATH: hf_cache_volume},
)
@modal.experimental.http_server(
    port=PORT,
    proxy_regions=[PROXY_REGION],
    exit_grace_period=15,
)
@modal.concurrent(target_inputs=TARGET_INPUTS)
class NemotronGenerator:
    """Serve Nemotron through SGLang's OpenAI-compatible HTTP API."""

    process: subprocess.Popen[str] | None = None

    @modal.enter()
    def start_server(self) -> None:
        command = [
            "sglang",
            "serve",
            "--model-path",
            MODEL_NAME,
            "--served-model-name",
            MODEL_NAME,
            "--host",
            "0.0.0.0",
            "--port",
            str(PORT),
            "--trust-remote-code",
            "--reasoning-parser",
            "nemotron_3",
            "--context-length",
            os.getenv("NEMOTRON_MODAL_CONTEXT_LENGTH", "32768"),
            "--chunked-prefill-size",
            os.getenv("NEMOTRON_MODAL_CHUNKED_PREFILL_SIZE", "8192"),
            "--mem-fraction-static",
            os.getenv("NEMOTRON_MODAL_MEM_FRACTION_STATIC", "0.82"),
        ]
        self.process = subprocess.Popen(command)
        _wait_ready(self.process)
        _warm_up_server()

    @modal.exit()
    def stop_server(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()


@app.local_entrypoint()
def main() -> None:
    print(
        "Deploy with `modal deploy modal_nemotron_service.py`. "
        "Then set NEMOTRON_BASE_URL to the deployed service URL plus `/v1`."
    )
