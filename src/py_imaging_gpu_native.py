"""
ctypes wrapper around cuda/imaging_gpu_native_lib.so — same SAR imaging kernel as
py_image_gpu.py / imaging_gpu.cu (cuda_image), without pycuda.

All tensor arguments must be CUDA float32 tensors (contiguous). Outputs are written in-place.

Future use: pass simulator channels + voxel grids directly on GPU into imaging_gpu_native_launch
to avoid CPU copies between simulation and ImageProcessor.generate_sar_image_simulation.

Build shared library (from repo root, adjust arch):

    nvcc --shared -Xcompiler -fPIC -arch sm_XX --std=c++14 \\
      -o src/data_processing/cuda/imaging_gpu_native_lib.so \\
      src/data_processing/cuda/imaging_gpu_native_lib.cu
"""

import ctypes
import os

import torch

_lib = None


def _get_lib():
    global _lib
    if _lib is None:
        so_path = os.path.join(
            os.path.dirname(__file__), "cuda", "imaging_gpu_native_lib.so"
        )
        _lib = ctypes.CDLL(so_path)
        _lib.imaging_gpu_native_launch.restype = ctypes.c_int
        _lib.imaging_gpu_native_launch.argtypes = [
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


def imaging_gpu_native_launch(
    out_r,
    out_i,
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
    samples_per_meas,
    start_ind=0,
    is_ti_radar=False,
    use_interpolated_processing=False,
    threads_per_block=512,
):
    """
    Launch cuda_image on device pointers.

    Layout matches py_image_gpu._run_cuda_image / imaging_gpu.cu:
      antenna_locs_flat: (num_antennas * 3,)
      meas_r, meas_i: flattened (num_antennas * samples_per_meas,)
      rx_offsets: (num_rx_antennas * 3,)
      out_r, out_i: (num_x * num_y * num_z,) — overwritten

    start_ind: voxel linear index offset (multi-GPU tiling); use 0 for full grid on one GPU.
    """
    lib = _get_lib()

    ret = lib.imaging_gpu_native_launch(
        _gpu_ptr(out_r),
        _gpu_ptr(out_i),
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
        ctypes.c_int(int(samples_per_meas)),
        ctypes.c_int(int(start_ind)),
        ctypes.c_int(1 if is_ti_radar else 0),
        ctypes.c_int(1 if use_interpolated_processing else 0),
        ctypes.c_int(int(threads_per_block)),
    )
    if ret != 0:
        raise RuntimeError(f"imaging_gpu_native_launch failed with code {ret}")
