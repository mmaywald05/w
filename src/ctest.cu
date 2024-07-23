#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

#define M_PI 3.14159265359

struct WAVHeader {
    char riff[4];        // "RIFF"
    int overall_size;    // File size minus 8 bytes
    char wave[4];        // "WAVE"
    char fmt_chunk_marker[4];  // "fmt "
    int length_of_fmt;   // Length of format data (usually 16)
    short format_type;   // Format type (1 is PCM)
    short channels;      // Number of channels
    int sample_rate;     // Sampling rate (blocks per second)
    int byterate;        // Bytes per second
    short block_align;   // 2=16-bit mono, 4=16-bit stereo
    short bits_per_sample; // Number of bits per sample
    char data_chunk_header[4]; // "data"
    int data_size;       // Size of data
};

__device__ int bitReverse(int n, int bits) {
    int reversed = 0;
    for (int i = 0; i < bits; ++i) {
        reversed = (reversed << 1) | (n & 1);
        n >>= 1;
    }
    return reversed;
}

__global__ void bitReversalKernel(float *real, float *imag, int n, int bits) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        int reversed = bitReverse(tid, bits);
        if (reversed > tid) {
            float tempReal = real[tid];
            float tempImag = imag[tid];
            real[tid] = real[reversed];
            imag[tid] = imag[reversed];
            real[reversed] = tempReal;
            imag[reversed] = tempImag;
        }
    }
}

__global__ void fftKernel(float *real, float *imag, int n, int step) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int m = n / 2;
    while (m >= step) {
        int j = tid % m;
        int k = (tid / m) * 2 * m + j;
        float angle = -2.0f * M_PI * j / (2.0f * m);
        float cosAngle = cosf(angle);
        float sinAngle = sinf(angle);
        float tempReal = cosAngle * real[k + m] - sinAngle * imag[k + m];
        float tempImag = sinAngle * real[k + m] + cosAngle * imag[k + m];
        real[k + m] = real[k] - tempReal;
        imag[k + m] = imag[k] - tempImag;
        real[k] += tempReal;
        imag[k] += tempImag;
        __syncthreads();
        m /= 2;
    }
}

void cudaFFT(float *real, float *imag, int n, int blockSize, int shift) {
    int bits = log2f(n);
    float *d_real, *d_imag;
    cudaMalloc(&d_real, n * sizeof(float));
    cudaMalloc(&d_imag, n * sizeof(float));
    cudaMemcpy(d_real, real, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_imag, imag, n * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid((n + blockSize - 1) / blockSize);
    dim3 block(blockSize);

    bitReversalKernel<<<grid, block>>>(d_real, d_imag, n, bits);
    cudaDeviceSynchronize();

    for (int step = 1; step < n; step *= 2) {
        fftKernel<<<grid, block>>>(d_real, d_imag, n, step);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(real, d_real, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(imag, d_imag, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_real);
    cudaFree(d_imag);
}

void readWavFile(const std::string &filePath, std::vector<float> &samples, int &sampleRate) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file: " << filePath << std::endl;
        exit(1);
    }

    WAVHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(WAVHeader));

    if (std::strncmp(header.riff, "RIFF", 4) != 0 || std::strncmp(header.wave, "WAVE", 4) != 0) {
        std::cerr << "Invalid WAV file format" << std::endl;
        exit(1);
    }

    sampleRate = header.sample_rate;
    int numSamples = header.data_size / (header.bits_per_sample / 8);
    samples.resize(numSamples);

    if (header.bits_per_sample == 16) {
        std::vector<short> tempSamples(numSamples);
        file.read(reinterpret_cast<char*>(tempSamples.data()), header.data_size);
        for (int i = 0; i < numSamples; ++i) {
            samples[i] = tempSamples[i] / 32768.0f;
        }
    } else if (header.bits_per_sample == 8) {
        std::vector<unsigned char> tempSamples(numSamples);
        file.read(reinterpret_cast<char*>(tempSamples.data()), header.data_size);
        for (int i = 0; i < numSamples; ++i) {
            samples[i] = tempSamples[i] / 128.0f - 1.0f;
        }
    } else {
        std::cerr << "Unsupported bit depth: " << header.bits_per_sample << std::endl;
        exit(1);
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path to wav file>" << std::endl;
        return 1;
    }

    std::string filePath = argv[1];
    std::vector<float> samples;
    int sampleRate;
    readWavFile(filePath, samples, sampleRate);

    int n = samples.size();
    int k = 1024;  // Block size for FFT
    int s = 64;    // Shift size for block

    std::vector<float> real(samples.begin(), samples.end());
    std::vector<float> imag(n, 0.0f);
    std::vector<float> frequencies(k);

    // Calculate frequency for each bin
    for (int i = 0; i < k; ++i) {
        frequencies[i] = i * sampleRate / k;
    }

    // Perform FFT blockwise
    for (int i = 0; i <= n - k; i += s) {
        cudaFFT(real.data() + i, imag.data() + i, k, 256, s);
    }



    // Print the output (for verification)
    for (int i = 0; i < k; ++i) {
        std::cout << "Bin " << i << " frequency: " << frequencies[i] << " Hz, real: " << real[i] << ", imag: " << imag[i] << std::endl;
    }

    return 0;
}