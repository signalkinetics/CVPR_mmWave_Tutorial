import numpy as np
import time 
c = 299792458
import gc
import torch

from src import py_imaging_gpu_native
from src import py_sn_est_gpu_native

class ImageProcessor:


    def generate_sar_image_simulation_gpu_native(self, received_signals, antenna_locations, image_coordinates):
        """
        Same SAR imaging as generate_sar_image_simulation for 77 GHz sim, using
        imaging_gpu_native_lib (ctypes) so channels and poses can stay on CUDA until
        the final image is copied to host.

        channels_gpu: (L, N, 1) or (L, N) complex or float tensor on CUDA
        antenna_locs_flat: length 3L float tensor (x,y,z per antenna row-major)
        locs_arrays: (x_locs, y_locs, z_locs) 1D voxel axes (tensor or array)

        antenna_batch_size:
            Max antennas per imaging kernel launch. FFT + measurement tensors are sized
            (batch, S) instead of (L, S), reducing peak VRAM while summing batches on GPU.
            None reads processing.gpu_native_antenna_batch_size; 0 means all antennas at once.
        """


        device = 'cuda'
        SAMPLES_PER_CHIRP = 32 #512 
        bandwidth = 3.000e9 
        max_rx = 2
        apply_ti_offset = True

        max_f = 80.50e9 
        min_f = 77.500e9
        slope = (max_f - min_f) / SAMPLES_PER_CHIRP
        wavelength = float(c / min_f)

        use_interpolated_processing = True 

        ch = received_signals#[:,::16,::2]
        print(ch.shape)
        if ch.dim() == 3: ch = ch.squeeze(-1)
        print(ch.shape)

        ch = ch[:, :SAMPLES_PER_CHIRP]
        L = int(ch.shape[0])

        if use_interpolated_processing:
            fft_sz = 32768 // 16  //2
            S = fft_sz
        else:
            S = int(ch.shape[1])

        fft_spacing = float(3e8 / (2 * bandwidth) * SAMPLES_PER_CHIRP / S)

        tx_offset = 0.005
        rx_offset = wavelength / 2
        rx_offsets = torch.tensor(
            
            [[-tx_offset - rx_offset * 3, 0, 0], 
            # [-tx_offset - rx_offset * 2, 0, 0], 
            [-tx_offset - rx_offset,     0, 0], 
            # [-tx_offset,                 0, 0]
            ],
            device=device,
            dtype=torch.float32,
        ).reshape(-1).contiguous()
 
        batch = 1024*16*2#4096 
        x_t, y_t, z_t = image_coordinates
        nx, ny, nz = len(x_t), len(y_t), len(z_t)

        nvox = nx * ny * nz
        acc_r = torch.zeros(nvox, device=device, dtype=torch.float32)
        acc_i = torch.zeros(nvox, device=device, dtype=torch.float32)
        out_r = torch.empty(nvox, device=device, dtype=torch.float32)
        out_i = torch.empty(nvox, device=device, dtype=torch.float32)

        # Compute image in batches (since we run out of GPU memory otherwise). 
        # To do so, compute image using a subset of antenna locations and sum image across batches (valid since image algorithm itself is linear combination across antennas)
        for s in range(0, L, batch):
            print(f'starting {s} out of {L}')
            torch.cuda.synchronize()
            t1 = time.time()
            e = min(s + batch, L)
            L_b = e - s
            ch_b = ch[s:e].to(device=device, dtype=torch.complex64).transpose(2,1).reshape((-1, SAMPLES_PER_CHIRP))#.reshape((L_b, -1))
            # print(ch_b.shape)
            if use_interpolated_processing: # Take over-interpolated FFT of channels to pass to imaging algorithm
                pad_cols = fft_sz - ch_b.shape[1]
                # print(L_b)
                # print(pad_cols)
                if pad_cols > 0:
                    pad_b = torch.zeros(L_b*max_rx, pad_cols, dtype=torch.complex64, device=device)
                    padded_b = torch.cat([ch_b, pad_b], dim=1).contiguous()
                else:
                    padded_b = ch_b
                measurement_b = torch.fft.fft(padded_b, dim=1)
                S_b = fft_sz
            else: # Use raw measurements
                measurement_b = ch_b
                S_b = int(ch_b.shape[1])
            torch.cuda.synchronize()
            t2 = time.time()
            # Prep for GPU
            meas_r_b = measurement_b.real.contiguous().float().reshape(-1)
            meas_i_b = measurement_b.imag.contiguous().float().reshape(-1)

            # Get subset of antenna locations
            ant_b = antenna_locations[s : e].contiguous().float().to(device=device)
            ant_b = torch.repeat_interleave(ant_b, max_rx, dim=0)
            ant_b = ant_b.reshape(-1)
            torch.cuda.synchronize()
            t3 = time.time()
            py_imaging_gpu_native.imaging_gpu_native_launch(
                out_r, out_i, x_t, y_t, z_t, ant_b, meas_r_b, meas_i_b, rx_offsets,
                float(slope), float(wavelength), float(fft_spacing),
                nx, ny, nz, L_b*max_rx, max_rx, S_b,
                start_ind=0, is_ti_radar=apply_ti_offset,
                use_interpolated_processing=use_interpolated_processing,
                threads_per_block=512,
            )
            torch.cuda.synchronize()
            t4 = time.time()
            print(out_i)
            print(out_r)
            print(f'Timing: {t2-t1} {t3-t2} {t4-t3}')
            # print(0/0)
            acc_r.add_(out_r)
            acc_i.add_(out_i)
            del measurement_b, meas_r_b, meas_i_b, ch_b, padded_b, pad_b

        image_c = torch.complex(acc_r, acc_i) / float(L)
        image = image_c.reshape(nx, ny, nz)
        return image

    def generate_mmwave_surface_normals_gpu_native(self, received_signals, antenna_locations, image_coordinates, sum_image, initial_filter_percent=0.05):
        """
        Same surface normal estimation as generate_mmwave_surface_normals for 77 GHz data, using
        surface_normal_est_native_lib (ctypes) so channels, SAR image, and poses can stay on CUDA.

        sum_image: (nx, ny, nz) complex SAR image from generate_sar_image_simulation_gpu_native
        """
        device = 'cuda'
        SAMPLES_PER_CHIRP = 32 
        bandwidth = 3.000e9 
        max_rx = 2
        apply_ti_offset = True

        max_f = 80.50e9 
        min_f = 77.500e9
        slope = (max_f - min_f) / SAMPLES_PER_CHIRP
        wavelength = float(c / min_f)

        use_interpolated_processing = True 

        ch = received_signals#[:,::16,::2]
        print(ch.shape)
        if ch.dim() == 3: ch = ch.squeeze(-1)
        print(ch.shape)

        ch = ch[:, :SAMPLES_PER_CHIRP]
        L = int(ch.shape[0])

        if use_interpolated_processing:
            fft_sz = 32768 // 16 //2  
            S = fft_sz
        else:
            S = int(ch.shape[1])

        fft_spacing = float(3e8 / (2 * bandwidth) * SAMPLES_PER_CHIRP / S)

        tx_offset = 0.005
        rx_offset = wavelength / 2
        rx_offsets = torch.tensor(
            
            [[-tx_offset - rx_offset * 3, 0, 0], 
            # [-tx_offset - rx_offset * 2, 0, 0], 
            [-tx_offset - rx_offset,     0, 0], 
            # [-tx_offset,                 0, 0]
            ],
            device=device,
            dtype=torch.float32,
        ).reshape(-1).contiguous()
 
        batch = 1024*16*2 #4096 
        x_t, y_t, z_t = image_coordinates
        nx, ny, nz = len(x_t), len(y_t), len(z_t)

        sum_image = sum_image.to(device=device)
        image_r = sum_image.real.reshape(-1).contiguous().float()
        image_i = sum_image.imag.reshape(-1).contiguous().float()

        filter_mask = torch.abs(sum_image) > (torch.max(torch.abs(sum_image)) * initial_filter_percent)
        valid_idx_x, valid_idx_y, valid_idx_z = torch.where(filter_mask)
        valid_idx_x = valid_idx_x.to(dtype=torch.int32)
        valid_idx_y = valid_idx_y.to(dtype=torch.int32)
        valid_idx_z = valid_idx_z.to(dtype=torch.int32)
        num_valid_idx = int(valid_idx_x.shape[0])

        acc_norm_x = torch.zeros(num_valid_idx, device=device, dtype=torch.float32)
        acc_norm_y = torch.zeros(num_valid_idx, device=device, dtype=torch.float32)
        acc_norm_z = torch.zeros(num_valid_idx, device=device, dtype=torch.float32)
        out_norm_x = torch.empty(num_valid_idx, device=device, dtype=torch.float32)
        out_norm_y = torch.empty(num_valid_idx, device=device, dtype=torch.float32)
        out_norm_z = torch.empty(num_valid_idx, device=device, dtype=torch.float32)

        for s in range(0, L, batch):
            print(f'starting {s} out of {L}')
            e = min(s + batch, L)
            L_b = e - s
            ch_b = ch[s:e].to(device=device, dtype=torch.complex64).transpose(2,1).reshape((-1, SAMPLES_PER_CHIRP))#.reshape((L_b, -1))
            if use_interpolated_processing: # Take over-interpolated FFT of channels to pass to imaging algorithm
                pad_cols = fft_sz - ch_b.shape[1]
                if pad_cols > 0:
                    pad_b = torch.zeros(L_b*max_rx, pad_cols, dtype=torch.complex64, device=device)
                    padded_b = torch.cat([ch_b, pad_b], dim=1).contiguous()
                else:
                    padded_b = ch_b
                measurement_b = torch.fft.fft(padded_b, dim=1)
                S_b = fft_sz
            else: # Use raw measurements
                measurement_b = ch_b
                S_b = int(ch_b.shape[1])

            meas_r_b = measurement_b.real.contiguous().float().reshape(-1)
            meas_i_b = measurement_b.imag.contiguous().float().reshape(-1)

            ant_b = antenna_locations[s : e].contiguous().float().to(device=device)
            ant_b = torch.repeat_interleave(ant_b, max_rx, dim=0)
            ant_b = ant_b.reshape(-1)

            py_sn_est_gpu_native.sn_est_gpu_native_launch(
                out_norm_x, out_norm_y, out_norm_z,
                valid_idx_x, valid_idx_y, valid_idx_z,
                image_r, image_i,
                x_t, y_t, z_t, ant_b, meas_r_b, meas_i_b, rx_offsets,
                float(slope), float(wavelength), float(fft_spacing),
                nx, ny, nz, L_b*max_rx, max_rx, num_valid_idx, S_b,
                start_ind=0, is_ti_radar=apply_ti_offset,
                threads_per_block=512,
            )
            acc_norm_x.add_(out_norm_x)
            acc_norm_y.add_(out_norm_y)
            acc_norm_z.add_(out_norm_z)
            del measurement_b, meas_r_b, meas_i_b, ch_b, padded_b, pad_b

        normals = torch.stack([acc_norm_x, acc_norm_y, acc_norm_z], dim=-1)
        all_normals = torch.full((nx, ny, nz, 3), float('nan'), device=device, dtype=torch.float32)
        all_normals[valid_idx_x, valid_idx_y, valid_idx_z] = normals
        all_normals /= torch.linalg.norm(all_normals, dim=-1, keepdim=True)
        return all_normals
 

