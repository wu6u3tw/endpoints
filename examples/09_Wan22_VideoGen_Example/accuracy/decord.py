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

"""Tiny decord shim for aarch64 where decord has no wheel.

Only covers the API surface VBench (commit 07bc8a4b…) actually uses:
    from decord import VideoReader, cpu
    decord.bridge.set_bridge("native"|"torch")
    VideoReader(path, num_threads=N[, width=W, height=H])
    len(vr)            -> int
    vr.get_avg_fps()   -> float
    vr.get_batch(idx)  -> object with .asnumpy()  (bridge=native)
                        or torch.Tensor (T,H,W,C) (bridge=torch)

Uses cv2 (opencv-python is already a vbench dep) so it's self-contained
and stable across torchvision versions — torchvision.io.read_video was
deprecated then removed in torchvision 0.20+, breaking the old shim.
"""

from __future__ import annotations

import cv2
import numpy as np

_BRIDGE = "native"


class _NativeBatch:
    """Wraps a (T,H,W,C) uint8 numpy array with .asnumpy() for native bridge."""

    def __init__(self, arr: np.ndarray):
        self._arr = arr

    def asnumpy(self):
        return self._arr

    @property
    def shape(self):
        return tuple(self._arr.shape)

    def __len__(self):
        return self._arr.shape[0]


class VideoReader:
    def __init__(
        self, path: str, num_threads: int = 1, width=None, height=None, ctx=None
    ):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise OSError(f"cannot open video: {path}")
        self._fps = float(cap.get(cv2.CAP_PROP_FPS))
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # cv2 reads BGR; vbench expects RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if width is not None and height is not None:
                frame = cv2.resize(
                    frame, (int(width), int(height)), interpolation=cv2.INTER_LINEAR
                )
            frames.append(frame)
        cap.release()
        if frames:
            self._frames = np.stack(frames, axis=0).astype(np.uint8)  # (T, H, W, C)
        else:
            self._frames = np.zeros((0, 0, 0, 3), dtype=np.uint8)

    def __len__(self):
        return int(self._frames.shape[0])

    def get_avg_fps(self):
        return self._fps

    def get_batch(self, indices):
        idx = list(indices)
        sel = self._frames[idx]  # (k, H, W, C) uint8
        if _BRIDGE == "torch":
            import torch

            return torch.from_numpy(sel)
        return _NativeBatch(sel)


class _Bridge:
    @staticmethod
    def set_bridge(name: str):
        global _BRIDGE
        _BRIDGE = name


bridge = _Bridge()


def cpu(_=0):
    return None
