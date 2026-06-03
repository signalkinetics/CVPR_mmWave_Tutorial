/*
 * SAR backprojection / imaging kernel — same logic as imaging_gpu.cu and the SourceModule
 * string in py_image_gpu.py (cuda_image), packaged as a shared library for ctypes (no pycuda).
 *
 * Host entry: imaging_gpu_native_launch — single-GPU full voxel grid (start_ind + grid cover voxels).
 *
 * Build (see setup.py):
 *   nvcc --shared -Xcompiler -fPIC -arch sm_XX --std=c++14 -o imaging_gpu_native_lib.so imaging_gpu_native_lib.cu
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

#define SPEED_LIGHT 2.99792458e8f
#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#define mult_r(a,b,c,d) ((a)*(c)-(b)*(d))
#define mult_i(a,b,c,d) ((a)*(d)+(b)*(c))

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        return -1; \
    } \
} while(0)


extern "C" __global__ void cuda_image(
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
    int SAMPLES_PER_MEAS,
    int start_ind,
    int is_ti_radar,
    int use_interpolated_processing)
{
    int ind = (blockIdx.x * blockDim.x + threadIdx.x) + start_ind;
    int z = ind % NUM_Z;
    ind /= NUM_Z;
    int y = ind % NUM_Y;
    ind /= NUM_Y;
    int x = ind % NUM_X;
    int lin = (blockIdx.x * blockDim.x + threadIdx.x) + start_ind;
    int total_vox = NUM_X * NUM_Y * NUM_Z;
    if (lin >= total_vox || x < 0 || y < 0 || z < 0 || x >= NUM_X || y >= NUM_Y || z >= NUM_Z)
        return;

    float x_loc = device_x_locs[x];
    float y_loc = device_y_locs[y];
    float z_loc = device_z_locs[z];

    float sum_r = 0.f;
    float sum_i = 0.f;

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
        float distance = forward_dist + back_dist;
        if (is_ti_radar != 0)
            distance += 0.15f;

        if (use_interpolated_processing != 0) {
            if (distance < 0.f || distance > fft_spacing * (float)SAMPLES_PER_MEAS)
                continue;

            int dist_bin = (int)floorf(distance / fft_spacing / 2.f);
            float real_meas = device_measurements_r[i * SAMPLES_PER_MEAS + dist_bin];
            float imag_meas = device_measurements_i[i * SAMPLES_PER_MEAS + dist_bin];
            float real_phase = cosf(-2.f * M_PI * distance / wavelength);
            float imag_phase = sinf(-2.f * M_PI * distance / wavelength);
            sum_r += mult_r(real_meas, imag_meas, real_phase, imag_phase);
            sum_i += mult_i(real_meas, imag_meas, real_phase, imag_phase);
        } else {
            for (unsigned int j = 0; j < (unsigned int)SAMPLES_PER_MEAS; j++) {
                float real_meas = device_measurements_r[i * SAMPLES_PER_MEAS + (int)j];
                float imag_meas = device_measurements_i[i * SAMPLES_PER_MEAS + (int)j];
                float ph = -2.f * M_PI * distance / wavelength +
                           (-2.f * M_PI * distance * (float)j * slope / SPEED_LIGHT);
                float real_phase = cosf(ph);
                float imag_phase = sinf(ph);
                sum_r += mult_r(real_meas, imag_meas, real_phase, imag_phase);
                sum_i += mult_i(real_meas, imag_meas, real_phase, imag_phase);
            }
        }
    }

    int out_idx = x * NUM_Y * NUM_Z + y * NUM_Z + z;
    device_p_xyz_r[out_idx] = sum_r;
    device_p_xyz_i[out_idx] = sum_i;
}


extern "C" int imaging_gpu_native_launch(
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
    int samples_per_meas,
    int start_ind,
    int is_ti_radar,
    int use_interpolated_processing,
    int threads_per_block)
{
    if (threads_per_block <= 0 || threads_per_block > 1024)
        return -3;
    long long total_vox = (long long)num_x * (long long)num_y * (long long)num_z;
    long long active = total_vox - (long long)start_ind;
    if (active <= 0)
        return 0;

    int grid_dim = (int)((active + (long long)threads_per_block - 1) / (long long)threads_per_block);

    dim3 block(threads_per_block, 1, 1);
    dim3 grid(grid_dim, 1, 1);

    cuda_image<<<grid, block>>>(
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
        samples_per_meas,
        start_ind,
        is_ti_radar,
        use_interpolated_processing);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}
