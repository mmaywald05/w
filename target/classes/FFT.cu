#include <cuda_runtime.h>
#include <math.h>

#define PI 3.1415926535897932384626433832795

__device__ void complexMul(float a_real, float a_imag, float b_real, float b_imag, float *c_real, float *c_imag) {
    *c_real = a_real * b_real - a_imag * b_imag;
    *c_imag = a_real * b_imag + a_imag * b_real;
}

__device__ void complexAdd(float a_real, float a_imag, float b_real, float b_imag, float *c_real, float *c_imag) {
    *c_real = a_real + b_real;
    *c_imag = a_imag + b_imag;
}

__device__ void complexSub(float a_real, float a_imag, float b_real, float b_imag, float *c_real, float *c_imag) {
    *c_real = a_real - b_real;
    *c_imag = a_imag - b_imag;
}

__device__ void exp_i(float theta, float *real, float *imag) {
    *real = cosf(theta);
    *imag = sinf(theta);
}

__global__ void fft_kernel(float* input_real, float* input_imag, float* output_real, float* output_imag, int N, int K, int totalSamples) {
    int blockId = blockIdx.x;
    int startIdx = blockId * K;

    if (startIdx + N <= totalSamples) {
        for (int m = 2; m <= N; m <<= 1) {
            int mh = m >> 1;
            for (int k = threadIdx.x; k < mh; k += blockDim.x) {
                float w_real, w_imag;
                exp_i(-2.0f * PI * k / m, &w_real, &w_imag);
                for (int i = startIdx + k; i < startIdx + N; i += m) {
                    int j = i + mh;
                    float t_real, t_imag;
                    complexMul(w_real, w_imag, input_real[j], input_imag[j], &t_real, &t_imag);
                    float u_real = input_real[i];
                    float u_imag = input_imag[i];
                    complexAdd(u_real, u_imag, t_real, t_imag, &input_real[i], &input_imag[i]);
                    complexSub(u_real, u_imag, t_real, t_imag, &input_real[j], &input_imag[j]);
                }
            }
            __syncthreads();
        }

        for (int i = startIdx + threadIdx.x; i < startIdx + N; i += blockDim.x) {
            output_real[i] = input_real[i];
            output_imag[i] = input_imag[i];
        }
    }
}
