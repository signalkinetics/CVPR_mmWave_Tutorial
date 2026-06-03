"""
ctypes wrapper around cuda/surface_normal_est_native_lib.so — same surface normal kernel as
py_sn_est_gpu.py / surface_normal_est.cu (cuda_sn_est), without pycuda.

All tensor arguments must be CUDA tensors (float32 or int32, contiguous). Outputs are written
in-place into d_normals_x/y/z for each valid voxel index.

Build shared library (from tutorial/src/cuda, adjust arch):

    nvcc --shared -Xcompiler -fPIC -arch sm_XX --std=c++14 \\
      -o surface_normal_est_native_lib.so surface_normal_est_native_lib.cu
"""

import ctypes
import os

import torch

_lib = None


def _get_lib():
    global _lib
    if _lib is None:
        so_path = os.path.join(
            os.path.dirname(__file__), "cuda", "surface_normal_est_native_lib.so"
        )
        _lib = ctypes.CDLL(so_path)
        _lib.sn_est_gpu_native_launch.restype = ctypes.c_int
        _lib.sn_est_gpu_native_launch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
    return _lib


def _gpu_ptr(t):
    assert t is not None and t.is_cuda, f"Expected CUDA tensor, got {t}"
    return ctypes.c_void_p(t.data_ptr())


def sn_est_gpu_native_launch(
    normals_x,
    normals_y,
    normals_z,
    valid_idx_x,
    valid_idx_y,
    valid_idx_z,
    image_r,
    image_i,
    x_locs,
    y_locs,
    z_locs,
    antenna_locs_flat,
    meas_r,
    meas_i,
    rx_offsets_flat,
    slope,
    wavelength,
    fft_spacing,
    num_x,
    num_y,
    num_z,
    num_antennas,
    num_rx_antennas,
    num_valid_idx,
    samples_per_meas,
    start_ind=0,
    is_ti_radar=False,
    threads_per_block=512,
):
    """
    Launch cuda_sn_est on device pointers.

    Layout matches py_sn_est_gpu.GPUThread / surface_normal_est.cu:
      valid_idx_x/y/z: (num_valid_idx,) int32 — voxel indices into the SAR grid
      image_r, image_i: (num_x * num_y * num_z,) float32 — flattened SAR image
      antenna_locs_flat: (num_antennas * 3,)
      meas_r, meas_i: flattened (num_antennas * samples_per_meas,)
      rx_offsets_flat: (num_rx_antennas * 3,)
      normals_x/y/z: (num_valid_idx,) — overwritten for indices [start_ind, num_valid_idx)

    start_ind: valid-voxel linear index offset (multi-GPU tiling); use 0 for full list on one GPU.
    """
    lib = _get_lib()

    ret = lib.sn_est_gpu_native_launch(
        _gpu_ptr(normals_x),
        _gpu_ptr(normals_y),
        _gpu_ptr(normals_z),
        _gpu_ptr(valid_idx_x),
        _gpu_ptr(valid_idx_y),
        _gpu_ptr(valid_idx_z),
        _gpu_ptr(image_r),
        _gpu_ptr(image_i),
        _gpu_ptr(x_locs),
        _gpu_ptr(y_locs),
        _gpu_ptr(z_locs),
        _gpu_ptr(antenna_locs_flat),
        _gpu_ptr(meas_r),
        _gpu_ptr(meas_i),
        _gpu_ptr(rx_offsets_flat),
        ctypes.c_float(float(slope)),
        ctypes.c_float(float(wavelength)),
        ctypes.c_float(float(fft_spacing)),
        ctypes.c_int(int(num_x)),
        ctypes.c_int(int(num_y)),
        ctypes.c_int(int(num_z)),
        ctypes.c_int(int(num_antennas)),
        ctypes.c_int(int(num_rx_antennas)),
        ctypes.c_int(int(num_valid_idx)),
        ctypes.c_int(int(samples_per_meas)),
        ctypes.c_int(int(start_ind)),
        ctypes.c_int(1 if is_ti_radar else 0),
        ctypes.c_int(int(threads_per_block)),
    )
    if ret != 0:
        raise RuntimeError(f"sn_est_gpu_native_launch failed with code {ret}")
