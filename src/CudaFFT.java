import javax.sound.sampled.*;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import static jcuda.driver.JCudaDriver.*;
import static jcuda.runtime.JCuda.cudaSetDevice;

import jcuda.runtime.JCuda;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;


public class CudaFFT {
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

    public static void executeFFTOnWavFile(String wavFilePath, int blockSize, int shift, float threshold) {
        // Read the WAV file and extract samples



        float[] samples = readWavFile(wavFilePath);

        if (samples == null) {
            System.err.println("Failed to read the WAV file.");
            return;
        }

        int N = samples.length;

        double[] h_input = new double[2*N];

        for (int i = 0; i < N; i++) {
            h_input[2*i] = samples[i]; // Real part
            h_input[2*i+1] = 0;          // Imaginary part
        }


        JCudaDriver.setExceptionsEnabled(true);
        cudaSetDevice(0);

        // Initialize the driver API
        JCudaDriver.cuInit(0);

        // Create the context
        CUcontext context = new CUcontext();
        CUdevice device = new CUdevice();
        JCudaDriver.cuDeviceGet(device, 0);
        JCudaDriver.cuCtxCreate(context, 0, device);

        // Load the PTX file
        CUmodule module = new CUmodule();
        String ptxFileName = "C:\\Users\\Morit\\what\\src\\test.ptx";
        JCudaDriver.cuModuleLoad(module, ptxFileName);

        // Obtain the function pointer to the kernel function
        CUfunction function = new CUfunction();
        JCudaDriver.cuModuleGetFunction(function, module, "fft");


        /*
        // TESTING

        JCudaDriver.setExceptionsEnabled(true);
        cudaSetDevice(0);

        // Initialize the driver API
        JCudaDriver.cuInit(0);

        // Create the context
        CUcontext context = new CUcontext();
        CUdevice device = new CUdevice();
        JCudaDriver.cuDeviceGet(device, 0);
        JCudaDriver.cuCtxCreate(context, 0, device);

        // Load the PTX file
        CUmodule module = new CUmodule();
        String ptxFileName = "C:\\Users\\Morit\\what\\src\\test.ptx";
        JCudaDriver.cuModuleLoad(module, ptxFileName);

        // Obtain the function pointer to the kernel function
        CUfunction function = new CUfunction();
        JCudaDriver.cuModuleGetFunction(function, module, "test");


        int testN = 100;
        double[] test_in = new double[testN];
        double[] test_output = new double[testN];
        Arrays.fill(test_in, 15d);

        Arrays.fill(test_output, 0);

        CUdeviceptr d_input = new CUdeviceptr();
        cuMemAlloc(d_input, testN * Sizeof.DOUBLE);
        CUdeviceptr d_output = new CUdeviceptr();
        cuMemAlloc(d_output, testN * Sizeof.DOUBLE);

        cuMemcpyHtoD(d_input, Pointer.to(test_in), testN*Sizeof.DOUBLE);
        cuMemcpyHtoD(d_output, Pointer.to(test_output), testN*Sizeof.DOUBLE);
        Pointer testParams = Pointer.to(
                Pointer.to(d_input),
                Pointer.to(d_output),
                Pointer.to(new int[]{testN})
        );
        cuLaunchKernel(function,
                1, 1, 1,      // Grid dimension
                testN, 1, 1, // Block dimension (launching a single block for the setup)
                0, null,   // Shared memory size and stream
                testParams, null // Kernel- and extra parameters
        );
        cuCtxSynchronize();

        // Copy the results back to the host
        cuMemcpyDtoH(Pointer.to(test_output), d_output, testN * Sizeof.DOUBLE);


        for (double b: test_output){
            System.out.print(b +" ");

        }

        cuMemFree(d_input);
        cuMemFree(d_output);
        JCudaDriver.cuCtxDestroy(context);


         */


        // Allocate memory on the device for the complex input/output data
        CUdeviceptr d_input = new CUdeviceptr();
        cuMemAlloc(d_input, 2*N * Sizeof.DOUBLE);

        // Allocate memory on the device for the amplitude results
        CUdeviceptr d_output = new CUdeviceptr();
        cuMemAlloc(d_output, N * Sizeof.DOUBLE);

        // Copy input data from host to device
        cuMemcpyHtoD(d_input, Pointer.to(h_input), 2*N * Sizeof.DOUBLE);

        // Set the execution parameters

        Pointer kernelParameters = Pointer.to(
                Pointer.to(d_input),
                Pointer.to(d_output),
                Pointer.to(new int[]{N}),
                Pointer.to(new int[]{blockSize}),
                Pointer.to(new int[]{shift})
        );



        // Launch the FFT and threshold kernel
        cuLaunchKernel(function,
                1, 1, 1,      // Grid dimension
                512, 1, 1, // Block dimension (launching a single block for the setup)
                0, null,   // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
        );
        cuCtxSynchronize();

        // Copy the results back to the host
        double[] h_output = new double[N];
        cuMemcpyDtoH(Pointer.to(h_output), d_output, N * Sizeof.DOUBLE);

        // Process results
        //displayResults(h_output);

        for(double f: h_output){
            System.out.print(f+" ");
        }

        // Clean up
        cuMemFree(d_input);
        cuMemFree(d_output);
        JCudaDriver.cuCtxDestroy(context);


    }
    public static void displayResults(double[] amplitudes){
        for (int i = 0; i < amplitudes.length; i++) {
            if (amplitudes[i] > 0) {
                System.out.printf("Bin %d: Amplitude = %f\n", i, amplitudes[i]);
            }
        }
    }


    public static float[] complexAbs(float[] real, float[] imag){
        if(real.length != imag.length){
            throw new IllegalArgumentException("real and imaginary parts must be same length. Something went wrong in the Kernel oh god pls no");
        }
        float[] result = new float[real.length];

        for (int i = 0; i < result.length; i++) {
            result[i] = real[i]*real[i] + imag[i]*imag[i];
        }

        return result;
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
