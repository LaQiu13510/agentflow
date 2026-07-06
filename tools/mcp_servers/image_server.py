"""Image generation MCP-style service."""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from config import (
    IMAGE_API_BASE,
    IMAGE_API_KEY,
    IMAGE_MODEL,
    IMAGE_OUTPUT_DIR,
    IMAGE_RETRY_COUNT,
    IMAGE_TIMEOUT_SECONDS,
)
from tools.mcp_base import LocalMCPServer, ToolResult, ToolSpec


class ImageMCPServer(LocalMCPServer):
    """Generate images with an OpenAI-compatible image API."""

    server_name = "image"

    def register_tools(self):
        self.add_tool(ToolSpec("health", "检查图片生成 API 配置"), self.health)
        self.add_tool(
            ToolSpec(
                "generate_image",
                "根据文本提示词生成图片，并保存到本地 data/generated_images",
                {"prompt": "图片提示词", "size": "1024x1024"},
            ),
            self.generate_image,
        )

    def health(self) -> ToolResult:
        if not IMAGE_API_KEY:
            return ToolResult(False, "IMAGE_API_KEY 未配置")
        detail = f"Image API configured, model={IMAGE_MODEL}"
        if IMAGE_API_BASE:
            detail += ", custom base URL enabled"
        return ToolResult(True, detail, {"model": IMAGE_MODEL})

    def generate_image(self, prompt: str, size: str = "1024x1024") -> ToolResult:
        if not IMAGE_API_KEY:
            return ToolResult(False, "IMAGE_API_KEY 未配置")
        if not prompt or not prompt.strip():
            return ToolResult(False, "prompt 不能为空")

        payload = {
            "model": IMAGE_MODEL,
            "prompt": prompt,
            "image": [],
            "size": size,
        }
        data = None
        last_error = ""
        for attempt in range(IMAGE_RETRY_COUNT + 1):
            try:
                data = self._post_generation(payload)
                break
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")[:500]
                last_error = f"HTTP {exc.code} {body}"
            except Exception as exc:
                last_error = str(exc)
            if attempt >= IMAGE_RETRY_COUNT:
                return ToolResult(False, f"图片 API 请求失败: {last_error}")

        item = (data.get("data") or [{}])[0]
        b64_json = item.get("b64_json")
        url = item.get("url")

        metadata = {"model": IMAGE_MODEL, "size": size}
        if b64_json:
            IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            image_path = self._write_png(b64_json)
            metadata["image_path"] = str(image_path)
            return ToolResult(
                True,
                f"图片已生成: {image_path}",
                metadata,
            )

        if url:
            metadata["image_url"] = url
            return ToolResult(True, f"图片已生成: {url}", metadata)

        return ToolResult(False, "图片 API 返回为空")

    def _post_generation(self, payload: dict) -> dict:
        base_url = (IMAGE_API_BASE or "https://www.right.codes/draw/v1").rstrip("/")
        if base_url.endswith("/v1"):
            endpoint = f"{base_url}/images/generations"
        else:
            endpoint = f"{base_url}/v1/images/generations"

        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {IMAGE_API_KEY}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=IMAGE_TIMEOUT_SECONDS) as response:
            body = response.read().decode("utf-8", errors="replace")
        return json.loads(body)

    def _write_png(self, b64_json: str) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = IMAGE_OUTPUT_DIR / f"generated-{timestamp}.png"
        path.write_bytes(base64.b64decode(b64_json))
        return path
