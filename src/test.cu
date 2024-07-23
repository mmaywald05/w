#include <stdlib.h>
#include <stdio.h>


#define PI 3.14159265359

#if __CUDA_ARCH__ < 600
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

extern "C"{
__device__ void bitReverse(double* a, int n) {
    int j = 0;
    for (int i = 1; i < n - 1; i++) {
        int bit = n >> 1;
        while (j >= bit) {
            j -= bit;
            bit >>= 1;
        }
        j += bit;
        if (i < j) {
            double temp = a[i];
            a[i] = a[j];
            a[j] = temp;
        }
    }
}

__global__ void test(double* input, double* output, int N, int blockSize, int shift, int blockNum){
  int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadId < blockNum+1){
    int startIndex = threadId * shift;
    int endIndex = startIndex + blockSize;

  }
}

__global__ void sharedTest(double* input, double* output, int N, int blockSize, int shift, int blocks){
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    int startIndex = threadId * shift;
    int endIndex = startIndex + blockSize;

    __shared__ float real[];
    __shared__ float imag[];

    for(int i =0; i < blockSize ; i++){
        real[i] = 5;
        imag[i] = 0;
    }
   __syncthreads();

    for(int i =0 ; i < blockSize;i++){
        imag[i] = real[i];
    }
   __syncthreads();


}


/*
input: Complex-like Array. Array of size 2*N where every even number is real, every odd is imaginary
output: Amplitude Array of size N
N: Sample Size. Half the Size of input Array
Block Size: Size of Block to be processed
Shift: After each block
*/
__global__ void fft(double* input, double*output, int N, int blockSize, int shift, int blockN){
    int threadId = blockIdx.x  * blockDim.x + threadIdx.x;

    if(threadId < blockN+1){
        int startIndex = threadId * shift;
        int endIndex = startIndex + blockSize;

        double* input_real = (double*) malloc(blockSize*sizeof(double));
        double* input_imaginary = (double*) malloc(blockSize*sizeof(double));
        for(int i = 0; i<blockSize; i++){
            if(i < N){
                input_real[i] = input[2*i];
                input_imaginary[i] = input[2*i+1];
            }
        }

        bitReverse(input_real,N);

        for(int len = 2; len < blockSize; len <<= 1){
            double angle = 2 * PI / len;
            double wlen_r = cos(angle);
            double wlen_i = sin(angle);

            for(int i = 0; i < blockSize; i += len){
                double w_r = 1;
                double w_i = 0;

                for(int j = 0; j < len/2 ; j++){
                    double tempr = w_r * input_real[i + j + len / 2] - w_i * input_imaginary[i + j + len / 2];
                    double tempi = w_r * input_imaginary[i + j + len / 2] + w_i * input_real[i + j + len / 2];
                    double xr = input_real[i + j];
                    double xi = input_imaginary[i + j];
                    input_real[i + j] = xr + tempr;
                    input_imaginary[i + j] = xi + tempi;
                    input_real[i + j + len / 2] = xr - tempr;
                    input_imaginary[i + j + len / 2] = xi - tempi;
                    double temp = w_r;
                    w_r = w_r * wlen_r - w_i * wlen_i;
                    w_i = temp * wlen_i + w_i * wlen_r;

                }
            }
        }


        for(int i =startIndex; i < endIndex; i++){
            output[i] += input_real[i] * input_real[i] + input_imaginary[i] * input_imaginary[i];
        }
    }
}

}