#include <cuda_runtime.h>
#include <math.h>

#define PI 3.14159265358979323846

extern "C" __global__ void fftKernel(float* d_real, float* d_imag, int n) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx >= n) return;

    // Bit reversal

    int j = 0;
    for (int i = 0; i < log2f((float)n); ++i) {
        j = (j << 1) | (idx & 1);
        idx >>= 1;
    }
    if (idx < j) {
        float temp_real = d_real[idx];
        float temp_imag = d_imag[idx];
        d_real[idx] = d_real[j];
        d_imag[idx] = d_imag[j];
        d_real[j] = temp_real;
        d_imag[j] = temp_imag;
    }

    // Compute the FFT
    for (int s = 2; s <= n; s <<= 1) {
        int m = s >> 1;
        float wm_real = cosf(-2 * PI / s);
        float wm_imag = sinf(-2 * PI / s);
        for (int k = idx; k < n; k += s) {
            float w_real = 1.0f;
            float w_imag = 0.0f;
            for (int j = 0; j < m; ++j) {
                int t = k + j + m;
                float u_real = d_real[k + j];
                float u_imag = d_imag[k + j];
                float v_real = w_real * d_real[t] - w_imag * d_imag[t];
                float v_imag = w_real * d_imag[t] + w_imag * d_real[t];
                d_real[k + j] = u_real + v_real;
                d_imag[k + j] = u_imag + v_imag;
                d_real[t] = u_real - v_real;
                d_imag[t] = u_imag - v_imag;
                float temp_real = w_real * wm_real - w_imag * wm_imag;
                w_imag = w_real * wm_imag + w_imag * wm_real;
                w_real = temp_real;
            }
        }
    }
}
