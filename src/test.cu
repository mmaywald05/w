
#define PI 3.14159265359

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

__global__ void test(double* input, double*output, int N, int blockSize, int shift){
      int threadId = blockIdx.x * blockDim.x + threadIdx.x;
      if (threadId <N){
            output[threadId] = input[threadId];
      }
}

/*
input: Complex-like Array. Array of size 2*N where every even number is real, every odd is imaginary
output: Amplitude Array of size N
N: Sample Size. Half the Size of input Array
Block Size: Size of Block to be processed
Shift: After each block
*/
__global__ void fft(double* input, double*output, int N, int blockSize, int shift){
    int threadId = blockIdx.x  * blockDim.x + threadIdx.x;

    if(threadId < N){
        int startIndex = threadId * shift;
        int endIndex = startIndex + blockSize-1 ;

        double* input_real = (double*) malloc(N*sizeof(double));
        double* input_imaginary = (double*) malloc(N*sizeof(double));
        for(int i = startIndex; i<endIndex; i++){
            input_real[i] = input[2*i];
            input_imaginary[i] = input[2*i+1];
        }

        bitReverse(input_real,N);

        for(int len = 2; len < N; len <<= 1){
            double angle = 2 * PI / len;
            double wlen_r = cos(angle);
            double wlen_i = sin(angle);

            for(int i = 0; i < N; i += len){
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