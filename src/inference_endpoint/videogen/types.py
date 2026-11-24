# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wire models for trtllm-serve POST /v1/videos/generations."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class VideoExtraParams(BaseModel):
    """`extra_params` of the request: the Wan 2.2 A14B settings trtllm-serve accepts.

    The server answers any other key with HTTP 400, so the model forbids them.
    """

    model_config = ConfigDict(extra="forbid")

    guidance_scale_2: float = Field(
        default=3.0,
        description="Secondary guidance scale for null-text CFG (MLPerf: 3.0).",
    )
    boundary_ratio: float = Field(
        default=0.875,
        description=(
            "Boundary ratio in [0, 1] that switches CFG from guidance_scale to "
            "guidance_scale_2 once the timestep crosses boundary_ratio * "
            "num_train_timesteps (MLPerf: 0.875)."
        ),
    )


class VideoPathRequest(BaseModel):
    """Request body for POST /v1/videos/generations.

    Field for field the trtllm-serve VideoGenerationRequest wire schema (it
    forbids unknown fields, HTTP 422), so the request is sent as dumped. All
    fields have MLPerf defaults so that only `prompt` is required from the
    dataset.

    `response_format` defaults to "path" (the server saves the video and
    returns its path); a dataset that sets `response_format="file"` gets the
    video file back instead (the adapter does not derive it from
    `benchmark_mode`).
    """

    prompt: str
    negative_prompt: str | None = Field(
        default=None,
        description=(
            "Text describing what to avoid. None means the field is omitted "
            "from the JSON payload so trtllm-serve can apply its model default. "
            "The bundled MLPerf prompts dataset carries the canonical negative "
            "prompt per row."
        ),
    )
    size: str = Field(
        default="1280x720",
        description="Frame size in 'WxH' format (MLPerf: 720p landscape).",
    )
    seconds: float = Field(
        default=5.0,
        description="Video duration. 81 frames @ ~16.2 fps = 5 s (MLPerf standard).",
    )
    fps: int = Field(default=16, description="Frames per second (MLPerf: 16).")
    num_inference_steps: int = Field(
        default=20, description="Denoising steps (MLPerf: 20)."
    )
    guidance_scale: float = Field(
        default=4.0, description="CFG guidance scale (MLPerf: 4.0)."
    )
    seed: int = Field(default=42, description="Random seed (MLPerf: 42).")
    num_frames: int = Field(
        default=81,
        description="Total video frames. 81 frames @ ~16.2 fps = 5 s (MLPerf: 81).",
    )
    extra_params: VideoExtraParams = Field(
        default_factory=lambda: VideoExtraParams(),
        description="Model-specific parameters (Wan 2.2 A14B's two-stage CFG).",
    )
    format: Literal["mp4", "avi", "auto"] = "auto"
    response_format: Literal["path", "file"] = "path"


class VideoPathResponse(BaseModel):
    """A path response (response_format="path"), after the adapter maps the envelope.

    The server saves the encoded video to its local filesystem (e.g. Lustre)
    and returns the absolute path instead of video bytes.
    Used in perf mode: MLPerf defines query completion as server finishing
    generation, so bytes do not need to cross the wire.

    `extra="forbid"` so a server returning unexpected fields (or both
    video_path and video_bytes) fails loudly at deserialisation.
    """

    model_config = ConfigDict(extra="forbid")

    video_id: str
    video_path: str


class VideoPayloadResponse(BaseModel):
    """Response body with the video inline as base64."""

    model_config = ConfigDict(extra="forbid")

    video_id: str
    video_bytes: str  # base64-encoded video content
