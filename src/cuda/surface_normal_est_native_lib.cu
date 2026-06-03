/*
 * mmWave surface normal estimation kernel — same logic as surface_normal_est.cu and the
 * SourceModule string in py_sn_est_gpu.py (cuda_sn_est), packaged as a shared library for
 * ctypes (no pycuda).
 *
 * Host entry: sn_est_gpu_native_launch — single-GPU launch over valid voxel indices.
 *
 * Build (see py_sn_est_gpu_native.py):
 *   nvcc --shared -Xcompiler -fPIC -arch sm_XX --std=c++14 \
 *     -o surface_normal_est_native_lib.so surface_normal_est_native_lib.cu
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#define mult_r(a, b, c, d) ((a) * (c) - (b) * (d))
#define mult_i(a, b, c, d) ((a) * (d) + (b) * (c))

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        return -1; \
    } \
} while(0)


extern "C" __global__ void cuda_sn_est(
    float* device_normals_x,
    float* device_normals_y,
    float* device_normals_z,
    int* valid_idx_x,
    int* valid_idx_y,
    int* valid_idx_z,
    float* device_p_xyz_r,
    float* device_p_xyz_i,
    float* device_x_locs,
    float* device_y_locs,
    float* device_z_locs,
    float* device_antenna_locs,
    float* device_measurements_r,
    float* device_measurements_i,
    float* rx_offsets,
    float slope,
    float wavelength,
    float fft_spacing,
    int NUM_X,
    int NUM_Y,
    int NUM_Z,
    int NUM_ANTENNAS,
    int NUM_RX_ANTENNAS,
    int NUM_VALID_IDX,
    int SAMPLES_PER_MEAS,
    int start_ind,
    int is_ti_radar)
{
    (void)slope;

    int ind = (blockIdx.x * blockDim.x + threadIdx.x) + start_ind;
    if (ind >= NUM_VALID_IDX || ind < 0)
        return;

    int x = valid_idx_x[ind];
    int y = valid_idx_y[ind];
    int z = valid_idx_z[ind];
    if (x < 0 || y < 0 || z < 0 || x >= NUM_X || y >= NUM_Y || z >= NUM_Z)
        return;

    float x_loc = device_x_locs[x];
    float y_loc = device_y_locs[y];
    float z_loc = device_z_locs[z];

    int out_idx = x * NUM_Y * NUM_Z + y * NUM_Z + z;
    float image_r = device_p_xyz_r[out_idx];
    float image_i = device_p_xyz_i[out_idx];
    float image_mag = sqrtf(image_r * image_r + image_i * image_i);
    if (image_mag <= 0.f)
        return;

    float sn_x = 0.f;
    float sn_y = 0.f;
    float sn_z = 0.f;

    float max_dist = fft_spacing * (float)SAMPLES_PER_MEAS;
    float ti_offset = (is_ti_radar != 0) ? 0.15f : 0.f;

    for (unsigned int i = 0; i < (unsigned int)NUM_ANTENNAS; i++) {
        float x_antenna_loc = device_antenna_locs[i * 3 + 0];
        float y_antenna_loc = device_antenna_locs[i * 3 + 1];
        float z_antenna_loc = device_antenna_locs[i * 3 + 2];
        float antenna_x_diff = x_loc - x_antenna_loc;
        float antenna_y_diff = y_loc - y_antenna_loc;
        float antenna_z_diff = z_loc - z_antenna_loc;

        int rx_num = i % NUM_RX_ANTENNAS;
        float rx_offset_x = rx_offsets[rx_num * 3 + 0];
        float rx_offset_y = rx_offsets[rx_num * 3 + 1];
        float rx_offset_z = rx_offsets[rx_num * 3 + 2];

        float forward_dist = sqrtf(antenna_x_diff * antenna_x_diff +
                                   antenna_y_diff * antenna_y_diff +
                                   antenna_z_diff * antenna_z_diff);
        float back_x = antenna_x_diff - rx_offset_x;
        float back_y = antenna_y_diff - rx_offset_y;
        float back_z = antenna_z_diff - rx_offset_z;
        float back_dist = sqrtf(back_x * back_x + back_y * back_y + back_z * back_z);
        float distance = forward_dist + back_dist + ti_offset;

        if (distance < 0.f || distance > max_dist)
            continue;

        int dist_bin = (int)floorf(distance / fft_spacing / 2.f);
        if (dist_bin < 0 || dist_bin >= SAMPLES_PER_MEAS)
            continue;

        float real_meas = device_measurements_r[i * SAMPLES_PER_MEAS + dist_bin];
        float imag_meas = device_measurements_i[i * SAMPLES_PER_MEAS + dist_bin];
        float real_phase = cosf(-2.f * M_PI * distance / wavelength);
        float imag_phase = sinf(-2.f * M_PI * distance / wavelength);
        float sum_r = mult_r(real_meas, imag_meas, real_phase, imag_phase);
        float sum_i = mult_i(real_meas, imag_meas, real_phase, imag_phase);

        float virtual_antenna_x = x_antenna_loc + rx_offset_x * 0.5f;
        float virtual_antenna_y = y_antenna_loc + rx_offset_y * 0.5f;
        float virtual_antenna_z = z_antenna_loc + rx_offset_z * 0.5f;

        float vec_x = virtual_antenna_x - x_loc;
        float vec_y = virtual_antenna_y - y_loc;
        float vec_z = virtual_antenna_z - z_loc;

        float weight = ((sum_r * image_r) + (sum_i * image_i)) / image_mag;

        sn_x += vec_x * weight;
        sn_y += vec_y * weight;
        sn_z += vec_z * weight;
    }

    device_normals_x[ind] = sn_x;
    device_normals_y[ind] = sn_y;
    device_normals_z[ind] = sn_z;
}


extern "C" int sn_est_gpu_native_launch(
    float* d_normals_x,
    float* d_normals_y,
    float* d_normals_z,
    int* d_valid_idx_x,
    int* d_valid_idx_y,
    int* d_valid_idx_z,
    float* d_p_xyz_r,
    float* d_p_xyz_i,
    float* d_x_locs,
    float* d_y_locs,
    float* d_z_locs,
    float* d_antenna_locs,
    float* d_meas_r,
    float* d_meas_i,
    float* d_rx_offsets,
    float slope,
    float wavelength,
    float fft_spacing,
    int num_x,
    int num_y,
    int num_z,
    int num_antennas,
    int num_rx_antennas,
    int num_valid_idx,
    int samples_per_meas,
    int start_ind,
    int is_ti_radar,
    int threads_per_block)
{
    if (threads_per_block <= 0 || threads_per_block > 1024)
        return -3;

    long long active = (long long)num_valid_idx - (long long)start_ind;
    if (active <= 0)
        return 0;

    int grid_dim = (int)((active + (long long)threads_per_block - 1) / (long long)threads_per_block);

    dim3 block(threads_per_block, 1, 1);
    dim3 grid(grid_dim, 1, 1);

    cuda_sn_est<<<grid, block>>>(
        d_normals_x,
        d_normals_y,
        d_normals_z,
        d_valid_idx_x,
        d_valid_idx_y,
        d_valid_idx_z,
        d_p_xyz_r,
        d_p_xyz_i,
        d_x_locs,
        d_y_locs,
        d_z_locs,
        d_antenna_locs,
        d_meas_r,
        d_meas_i,
        d_rx_offsets,
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
        start_ind,
        is_ti_radar);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}
