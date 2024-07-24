#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <chrono>
#include <thrust/complex.h>


using namespace std::chrono;
#define M_PI 3.14159265359

typedef struct {
    char    ChunkID[4];
    int32_t ChunkSize;
    char    Format[4];
    char    Subchunk1ID[4];
    int32_t Subchunk1Size;
    int16_t AudioFormat;
    int16_t NumChannels;
    int32_t SampleRate;
    int32_t ByteRate;
    int16_t BlockAlign;
    int16_t BitsPerSample;
    char    Subchunk2ID[4];
    int32_t Subchunk2Size;
} WavHeader;

// Define a complex number type
typedef float2 Complex;
__global__ void dftKernel(const Complex* input, Complex* output, int N, int k, int s, int numBlocks) {
    int tid = threadIdx.x;  // Index within the block (frequency bin)

    if (tid < k) {
        Complex sum = make_float2(0.0f, 0.0f);

        for (int b = 0; b < numBlocks; ++b) {
            Complex tempSum = make_float2(0.0f, 0.0f);
            for (int n = 0; n < k; ++n) {
                int index = b * s + n;
                if (index < N) {
                    float angle = 2.0f * M_PI * tid * n / k;
                    float cosAngle = cosf(angle);
                    float sinAngle = -sinf(angle);  // Note the negative sign for the DFT

                    tempSum.x += input[index].x * cosAngle - input[index].y * sinAngle;
                    tempSum.y += input[index].x * sinAngle + input[index].y * cosAngle;
                }
            }
            sum.x += tempSum.x;
            sum.y += tempSum.y;
        }
        output[tid] = make_float2(sum.x / numBlocks, sum.y / numBlocks);
    }
}

__device__ __host__ Complex make_complex(float real, float imag) {
    Complex c;
    c.x = real;
    c.y = imag;
    return c;
}

__device__ __host__ Complex complex_add(const Complex& a, const Complex& b) {
    return make_complex(a.x + b.x, a.y + b.y);
}

__device__ __host__ Complex complex_mul(const Complex& a, const Complex& b) {
    return make_complex(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__device__ __host__ float complex_mag(const Complex& c) {
    return sqrtf(c.x * c.x + c.y * c.y);
}


__global__ void mydftkernel(const Complex* input, float* magnitudes, int N, int k, int s, int numBlocks){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int startIndex = tid * s;
    int endIndex = startIndex + k;

    if(tid < numBlocks){
        for(int i = startIndex; i < endIndex; ++i){
            Complex number = make_complex(0,0);
            for(int j = startIndex; j < endIndex; ++j){
                double angle = 2 * M_PI * i * j / k;
                Complex w = make_complex(cosf(angle), -sinf(angle));
                Complex prod = complex_mul(input[j], w);
                number = complex_add(number, prod);

                // das hier ist kritisch, ich glaube das geht so nicht, besser die magnituden hier ausrechnen und einfach so übergeben
            }
            float mag = complex_mag(number)/numBlocks;
             // Print the magnitude before it is added to the magnitudes array for debugging
            atomicAdd(&magnitudes[(i-startIndex)], mag);
        }
    }
}

void computeDFTBlocks(const Complex* h_input, float* h_magnitudes, int N, int k, int s) {
    int numBlocks = (N - k) / s + 1;  // Calculate the number of blocks

    Complex* d_input;

    float* d_magnitudes;



    // Allocate device memory
    cudaMalloc((void**)&d_input, N * sizeof(Complex));

    cudaMalloc((void**)&d_magnitudes, k * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, h_input, N * sizeof(Complex), cudaMemcpyHostToDevice);
    cudaMemset(d_magnitudes, 0, k * sizeof(float));
    // Launch the DFT kernel with enough blocks and threads to cover all frequency bins



    mydftkernel<<<1024, 1024>>>(d_input, d_magnitudes, N, k, s, numBlocks);
    cudaDeviceSynchronize();

    // Launch the magnitude kernel
    //magnitudeKernel<<<1, blockSize>>>(d_output, d_magnitudes, k);

    // Copy the results back to the host
    cudaMemcpy(h_magnitudes, d_magnitudes, k * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_magnitudes);
}



void saveArrayToFile(const float* values, int numSamples, const std::string& filename) {
    std::ofstream outFile(filename); // Create an output file stream

    if (!outFile) {
        std::cerr << "Error: Could not open file for writing." << std::endl;
        return;
    }

    for (int i = 0; i < numSamples; ++i) {
        outFile << values[i] << std::endl; // Write each value on a new line
    }

    outFile.close(); // Close the file stream
}

std::vector<float> readWav(const std::string& filePath, int &sampleRate) {
      std::ifstream file(filePath, std::ios::binary);
      WavHeader header;

      if (!file.read(reinterpret_cast<char*>(&header), sizeof(WavHeader))) {
          throw std::runtime_error("Failed to read WAV file header.");
      }

      if (strncmp(header.ChunkID, "RIFF", 4) != 0 || strncmp(header.Format, "WAVE", 4) != 0) {
          throw std::runtime_error("File is not a valid WAV file.");
      }

      if (header.AudioFormat != 1) {
          throw std::runtime_error("Unsupported audio format. Only PCM format supported.");
      }

      if (header.BitsPerSample != 16) {
          throw std::runtime_error("Unsupported sample size. Only 16-bit samples supported.");
      }

      int sampleCount = header.Subchunk2Size / sizeof(int16_t);
      sampleRate = header.SampleRate;

    sampleCount = header.Subchunk2Size / sizeof(int16_t);
    std::vector<int16_t> buffer(sampleCount);

       file.read(reinterpret_cast<char*>(buffer.data()), header.Subchunk2Size);

      file.close();

      std::vector<float> samples(buffer.size());

    // Convert int16_t samples to float and normalize to [-1, 1]
     float maxAbsValue = 0;
     for (size_t i = 0; i < buffer.size(); ++i) {
        samples[i] = static_cast<float>(buffer[i]) / std::numeric_limits<int16_t>::max();
        if (std::abs(samples[i]) > maxAbsValue) {
            maxAbsValue = std::abs(samples[i]);
        }
     }

    if (maxAbsValue > 0) {
        for (size_t i = 0; i < sampleCount; ++i) {
            samples[i] /= maxAbsValue;
        }
    }

    return samples;
}


int main(int argc, char *argv[]) {
    auto start = high_resolution_clock::now();
    std::string filePath = argv[1];

    int sampleRate;
    std::vector<float> samples = readWav(filePath, sampleRate);

    //readWavFile(filePath, samples, sampleRate);
    int N = samples.size();;  // Number of source file samples
    int k = 512;    // blocksize
    int s = 64;     // shift
    int numBlocks = (N - k) / s + 1;


    Complex* h_input = (Complex*)malloc(N * sizeof(Complex));
    float* h_magnitudes = (float*)malloc(k * sizeof(float));

    // Initialize input data (example: sine wave)
    for (int n = 0; n < N; ++n) {
        h_input[n].x = samples[n];
        h_input[n].y = 0.0f;

    }


    std::cout << "Starting DFT...";
    // Compute the DFT
    computeDFTBlocks(h_input, h_magnitudes, N, k, s);
    std::cout << "done:" <<std::endl;

    // Print the magnitudes of the frequency bins
    std::cout << "k = Blocksize = " << k << std::endl;




    int pos, neg,zero;
    float max = 0;
    float min = FLT_MAX;
    int maxIndex = 0;
    int minIndex = 0;

    pos =0; neg=0;zero=0;
    for(int i = 0; i < k; ++i){
        h_magnitudes[i] = h_magnitudes[i] / numBlocks;

        if(h_magnitudes[i] > 0){
            ++pos;
        }else if (h_magnitudes[i] < 0){
            ++neg;
        }else{
            ++zero;
        }

        if(h_magnitudes[i]  > max){
            max = h_magnitudes[i];
            maxIndex = i;
        }
        if(h_magnitudes[i]< min ){
            minIndex = i;
            min = h_magnitudes[i];
        }


    }
    for(int i =0;  i< k; ++i){
        h_magnitudes[i] = (h_magnitudes[i]-min)/(max-min);
    }


    saveArrayToFile(h_magnitudes, k, "magnitudes.txt");

/*
    for(int i = 0; i < k;  ++i){
        h_magnitudes[i] = std::round(h_magnitudes[i]*100)/100;
    }
    */

    for(int i = 0; i < k; ++i){
        double frequency = i * sampleRate / k;
        printf("Frequency %f: Magnitude = %f\n", frequency, h_magnitudes[i]);
    }

    // Free host memory
    free(h_input);
    free(h_magnitudes);


    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout <<"GPU DFT took "<< duration.count()/1000 << "ms." << std::endl;


    return 0;
}
