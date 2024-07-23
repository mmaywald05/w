#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <chrono>

using namespace std::chrono;
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

// Define a complex number type
typedef float2 Complex;
__global__ void dftKernel(const Complex* input, Complex* output, float* magnitudes, int N) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;  // Frequency bin index

    if (k < N) {
        Complex sum = make_float2(0.0f, 0.0f);

        for (int n = 0; n < N; ++n) {
            float angle = 2.0f * M_PI * k * n / N;
            float cosAngle = cosf(angle);
            float sinAngle = -sinf(angle);  // Note the negative sign for the DFT

            sum.x += input[n].x * cosAngle - input[n].y * sinAngle;
            sum.y += input[n].x * sinAngle + input[n].y * cosAngle;
        }

        output[k] = sum;
        magnitudes[k] = sqrtf(sum.x * sum.x + sum.y * sum.y);
    }
}

void computeDFT(const Complex* h_input, Complex* h_output, float* h_magnitudes, int N) {
    Complex* d_input;
    Complex* d_output;
    float* d_magnitudes;

    // Allocate device memory
    cudaMalloc((void**)&d_input, N * sizeof(Complex));
    cudaMalloc((void**)&d_output, N * sizeof(Complex));
    cudaMalloc((void**)&d_magnitudes, N * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, h_input, N * sizeof(Complex), cudaMemcpyHostToDevice);

    // Launch the kernel with enough blocks to cover all frequency bins
    int blockSize = 1024;
    int numBlocks = (N + blockSize - 1) / blockSize;

    dftKernel<<<numBlocks, blockSize>>>(d_input, d_output, d_magnitudes, N);

    // Copy the results back to the host
    cudaMemcpy(h_output, d_output, N * sizeof(Complex), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_magnitudes, d_magnitudes, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_magnitudes);
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

    auto start = high_resolution_clock::now();
    std::string filePath = argv[1];
    std::vector<float> samples;
    int sampleRate;
    readWavFile(filePath, samples, sampleRate);

    int N = samples.size();;  // Size of the DFT



    Complex* h_input = (Complex*)malloc(N * sizeof(Complex));
    Complex* h_output = (Complex*)malloc(N * sizeof(Complex));
    float* h_magnitudes = (float*)malloc(N * sizeof(float));

    // Initialize input data (example: sine wave)
    for (int n = 0; n < N; ++n) {
        h_input[n].x = samples[n];
        h_input[n].y = 0.0f;
    }


    std::cout << "Starting DFT...";
    // Compute the DFT
    computeDFT(h_input, h_output, h_magnitudes, N);
    std::cout << "done:" <<std::endl;

    // Print the magnitudes of the frequency bins
    for (int k = 0; k < N; ++k) {
        printf("Frequency bin %d: Magnitude = %f\n", k, h_magnitudes[k]);
    }

    // Free host memory
    free(h_input);
    free(h_output);
    free(h_magnitudes);


    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout <<"CUDA FFT took "<< duration.count()/1000 << "ms." << std::endl;

    return 0;
}
