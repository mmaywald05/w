import javax.sound.sampled.*;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;

import java.io.File;
import java.io.IOException;

import static jcuda.driver.JCudaDriver.*;

public class JCudaExample {
    public static void main(String[] args) {
        String wavFilePath = "Fanfare60.wav";
        int blockSize = 1024;
        int offset = 32;
        float threshold = 0.0f;
        long start = System.currentTimeMillis();
        executeFFTOnWavFile(wavFilePath, blockSize, offset, threshold);
        long end = System.currentTimeMillis();
        System.out.println("Cuda Execution took " + (end-start) + " ms.");
    }

    public static void executeFFTOnWavFile(String wavFilePath, int blockSize, int offset, float threshold) {
        // Read the WAV file and extract samples
        float[] samples = readWavFile(wavFilePath);
        if (samples == null) {
            System.err.println("Failed to read the WAV file.");
            return;
        }

        // Initialize JCuda
        JCudaDriver.setExceptionsEnabled(true);
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);
        CUmodule module = new CUmodule();
        cuModuleLoad(module, "src/fftKernel.ptx");
        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module, "fftKernel");

        int blockSizeX = 256;

        int counter = 0;
        // Process the samples in blocks
        int numBlock = (samples.length-blockSize) / offset;

        for (int i = 0; i < samples.length - blockSize; i += offset) {
            //System.out.println("Processing Block " + ++counter + " of " + numBlock);
            float[] block = new float[2 * blockSize];
            for (int j = 0; j < blockSize; j++) {
                block[2 * j] = samples[i + j]; // Real part
                block[2 * j + 1] = 0; // Imaginary part
            }

            // Allocate device memory and copy data to the device
            CUdeviceptr deviceData = new CUdeviceptr();
            cuMemAlloc(deviceData, 2 * blockSize * Sizeof.FLOAT);
            cuMemcpyHtoD(deviceData, Pointer.to(block), 2 * blockSize * Sizeof.FLOAT);

            // Set up the kernel parameters
            Pointer kernelParameters = Pointer.to(
                    Pointer.to(deviceData),
                    Pointer.to(new int[]{blockSize}),
                    Pointer.to(new int[]{1}) // Initial step is 1
            );

            // Launch the kernel
            int gridSizeX = (int) Math.ceil((double) blockSize / blockSizeX);
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,     // Grid dimension
                    blockSizeX, 1, 1,    // Block dimension
                    0, null,             // Shared memory size and stream
                    kernelParameters, null // Kernel parameters
            );

            cuCtxSynchronize();


            // Copy the result back to the host
            cuMemcpyDtoH(Pointer.to(block), deviceData, 2 * blockSize * Sizeof.FLOAT);


            // Print the amplitudes above the threshold

            /*
            for (int j = 0; j < blockSize; j++) {
                float real = block[2 * j];
                float imag = block[2 * j + 1];
                float amplitude = (float) Math.sqrt(real * real + imag * imag);
                if (amplitude > threshold) {
                    System.out.println("Frequency " + j + ": Amplitude " + amplitude);
                }
            }

             */

            // Clean up

            cuMemFree(deviceData);
        }

        cuCtxDestroy(context);
    }

    private static float[] readWavFile(String filePath) {
        try {
            AudioInputStream audioInputStream = AudioSystem.getAudioInputStream(new File(filePath));
            AudioFormat format = audioInputStream.getFormat();
            byte[] audioBytes = FFTFactory.readAllBytes(audioInputStream);
            audioInputStream.close();

            int sampleSizeInBytes = format.getSampleSizeInBits() / 8;
            int numSamples = audioBytes.length / sampleSizeInBytes;
            float[] samples = new float[numSamples];
            for (int i = 0; i < numSamples; i++) {
                samples[i] = bytesToSample(audioBytes, i * sampleSizeInBytes, sampleSizeInBytes, format.isBigEndian());
            }
            return samples;
        } catch (UnsupportedAudioFileException | IOException e) {
            e.printStackTrace();
            return null;
        }
    }

    private static float bytesToSample(byte[] audioBytes, int offset, int sampleSizeInBytes, boolean bigEndian) {
        int sample = 0;
        if (bigEndian) {
            for (int i = 0; i < sampleSizeInBytes; i++) {
                sample |= (audioBytes[offset + i] & 0xFF) << (8 * (sampleSizeInBytes - 1 - i));
            }
        } else {
            for (int i = 0; i < sampleSizeInBytes; i++) {
                sample |= (audioBytes[offset + i] & 0xFF) << (8 * i);
            }
        }
        return sample / (float) (1 << (8 * sampleSizeInBytes - 1));
    }
}
