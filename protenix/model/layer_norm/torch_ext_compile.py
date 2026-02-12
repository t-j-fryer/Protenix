# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import warnings
from typing import Any

from torch.utils.cpp_extension import load


_ARCH_TO_GENCODE = {
    "7.0": "arch=compute_70,code=sm_70",
    "8.0": "arch=compute_80,code=sm_80",
    "8.6": "arch=compute_86,code=sm_86",
    "9.0": "arch=compute_90,code=sm_90",
    "12.0": "arch=compute_120,code=sm_120",
}


def _resolve_cuda_arch_list() -> list[str]:
    """Resolve CUDA architectures from TORCH_CUDA_ARCH_LIST.

    Supports values like ``12.0`` and ``12.0+PTX`` and falls back to the
    legacy default list when no supported architecture is found.
    """

    raw_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "7.0;8.0")
    parsed_arches: list[str] = []

    for arch in raw_arch_list.split(";"):
        normalized_arch = arch.strip().replace("+PTX", "")
        if normalized_arch in _ARCH_TO_GENCODE:
            parsed_arches.append(normalized_arch)

    if parsed_arches:
        return parsed_arches

    warnings.warn(
        "No supported architecture found in TORCH_CUDA_ARCH_LIST="
        f"{raw_arch_list!r}. Falling back to '7.0;8.0'.",
        stacklevel=2,
    )
    return ["7.0", "8.0"]


def _resolve_extra_cuda_cflags() -> list[str]:
    extra_cuda_cflags = [
        "-O3",
        "--use_fast_math",
        "-DVERSION_GE_1_1",
        "-DVERSION_GE_1_3",
        "-DVERSION_GE_1_5",
        "-std=c++17",
        "-maxrregcount=32",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
    ]

    for arch in _resolve_cuda_arch_list():
        extra_cuda_cflags.extend(["-gencode", _ARCH_TO_GENCODE[arch]])

    return extra_cuda_cflags


def compile(
    name: str, sources: list[str], extra_include_paths: list[str], build_directory: str
) -> Any:
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0;8.0")
    return load(
        name=name,
        sources=sources,
        extra_include_paths=extra_include_paths,
        extra_cflags=[
            "-O3",
            "-DVERSION_GE_1_1",
            "-DVERSION_GE_1_3",
            "-DVERSION_GE_1_5",
        ],
        extra_cuda_cflags=_resolve_extra_cuda_cflags(),
        verbose=True,
        build_directory=build_directory,
    )
