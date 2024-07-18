#include <cufft.h>
#include <cuda_runtime.h>

extern "C" {

    __global__ void apply_threshold(cufftComplex* data, float* amplitude, int n, float threshold) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            float re = data[idx].x;
            float im = data[idx].y;
            float amp = sqrtf(re * re + im * im);
            amplitude[idx] = amp > threshold ? amp : 0.0f;
        }
    }

    extern void fft(cufftComplex* data, float* amplitude, int n, float threshold, int blockSize) {
        // Create cuFFT plan
        cufftHandle plan;
        cufftPlan1d(&plan, n, CUFFT_C2C, 1);

        // Execute FFT
        cufftExecC2C(plan, data, data, CUFFT_FORWARD);

        // Launch the kernel to apply the threshold
        int numBlocks = (n + blockSize - 1) / blockSize;
        apply_threshold<<<numBlocks, blockSize>>>(data, amplitude, n, threshold);

        // Destroy cuFFT plan
        cufftDestroy(plan);

        // Wait for GPU to finish
        cudaDeviceSynchronize();
    }
}
