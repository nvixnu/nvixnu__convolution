#include "nvixnu__convolution.h"

void nvixnu__1d_convolution_host(double *input, double *output, const int length, const double *mask, const int mask_width){
	int ghosts_by_side = mask_width/2;
	double sum;
	int input_idx;

	for(int out_idx = 0; out_idx < length; out_idx++){ // Iterates through each output position to calculate it
		sum = 0;
		for(int mask_idx = 0; mask_idx < mask_width; mask_idx++){ // Iterates through each mask position
			input_idx = out_idx - ghosts_by_side + mask_idx; // Calculates the input index
			if(input_idx >= 0 && input_idx < length){ // Check if the input index is not a ghost
				sum+=input[input_idx]*mask[mask_idx]; //Performs the convolution
			}
		}
		output[out_idx] = sum;
	}
}

__global__
void nvixnu__1d_convolution_kernel(double *input, double *output, const int length, double *mask, const int mask_width){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	extern __shared__ double shared[];

	// Each thread loads data from global to the block shared memory
	shared[threadIdx.x] = tid < length ? input[tid] : 0.0;
	__syncthreads();

	// Defines the data index that belongs to each tile
	int this_tile_start_point = blockIdx.x * blockDim.x;
	int next_tile_start_point = (blockIdx.x + 1) * blockDim.x;

	// Go back int(mask_width/2) positions in order to start from the block scope external cells (halos or ghosts placed before the this_tile_start_point position)
	int n_start_point = tid - (mask_width/2);
	double p = 0;

	for(int j = 0; j < mask_width; j++){

		int n_index = n_start_point + j;
		if(n_index >= 0 && n_index < length){ //Check if the n_index not refers to a ghost cell
			if(n_index >= this_tile_start_point && n_index < next_tile_start_point){ // If is an internal cell (true) or a halo cell (false)
				p += shared[threadIdx.x + j - mask_width/2]*mask[j];
			}else{
				p += input[n_index] * mask[j]; //Takes the N[value] from the cache (Luckily!) or from the global memory and performs the convolution
			}
		}
	}
	output[tid] = p;
}
