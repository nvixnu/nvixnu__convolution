#ifndef NVIXNU__CONVOLUTION_H_
#define NVIXNU__CONVOLUTION_H_

/**
* Kernel that performs 1D convolution operation
* @param input The input array
* @param output The output array
* @param length The input array length
* @param mask The convolution mask
* @param mask_width The width of the convolution mask (kernel)
*/
__global__ void nvixnu__1d_convolution_kernel(double *input, double *output, const int length, double *mask, const int mask_width);

/**
* The host function that performs 1D convolution operation
* @param input The input array
* @param output The output array
* @param length The input array length
* @param mask The convolution mask
* @param mask_width The width of the convolution mask (kernel)
*/
void nvixnu__1d_convolution_host(double *input, double *output, const int length, const double *mask, const int mask_width);

#endif /* NVIXNU__CONVOLUTION_H_ */
