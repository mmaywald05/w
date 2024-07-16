import javax.sound.sampled.*;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import java.io.File;
import java.io.IOException;


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

    public static void executeFFTOnWavFile(String wavFilePath, int N, int K, float threshold) {
        // Read the WAV file and extract samples
        float[] samples = readWavFile(wavFilePath);

        if (samples == null) {
            System.err.println("Failed to read the WAV file.");
            return;
        }
        int totalSamples = samples.length;

        float[] input_real = new float[totalSamples];
        float[] input_imag = new float[totalSamples];
        for (int i = 0; i < totalSamples; i++) {
            input_real[i] = samples[i]; // Real part
            input_imag[i] = 0;          // Imaginary part
        }

        JCudaDriver.cuInit(0);
        CUdevice device = new CUdevice();
        JCudaDriver.cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        JCudaDriver.cuCtxCreate(context, 0, device);

        // Load the PTX file
        CUmodule module = new CUmodule();
        JCudaDriver.cuModuleLoad(module, "FFT.ptx");

        // Obtain the function pointer to the kernel function
        CUfunction function = new CUfunction();
        JCudaDriver.cuModuleGetFunction(function, module, "fft_kernel");

        // Allocate memory on the GPU
        CUdeviceptr d_input_real = new CUdeviceptr();
        CUdeviceptr d_input_imag = new CUdeviceptr();
        CUdeviceptr d_output_real = new CUdeviceptr();
        CUdeviceptr d_output_imag = new CUdeviceptr();
        JCudaDriver.cuMemAlloc(d_input_real, totalSamples * Sizeof.FLOAT);
        JCudaDriver.cuMemAlloc(d_input_imag, totalSamples * Sizeof.FLOAT);
        JCudaDriver.cuMemAlloc(d_output_real, totalSamples * Sizeof.FLOAT);
        JCudaDriver.cuMemAlloc(d_output_imag, totalSamples * Sizeof.FLOAT);

        // Copy the input data from host to device
        JCudaDriver.cuMemcpyHtoD(d_input_real, Pointer.to(input_real), totalSamples * Sizeof.FLOAT);
        JCudaDriver.cuMemcpyHtoD(d_input_imag, Pointer.to(input_imag), totalSamples * Sizeof.FLOAT);


        // Number of blocks to process
        int numBlocks = (totalSamples - N) / K + 1;

        Pointer kernelParameters = Pointer.to(
                Pointer.to(d_input_real),
                Pointer.to(d_input_imag),
                Pointer.to(d_output_real),
                Pointer.to(d_output_imag),
                Pointer.to(new int[]{N}),
                Pointer.to(new int[]{K}),
                Pointer.to(new int[]{totalSamples})
        );

        // Launch the kernel
        int blockSize = 256;
        JCudaDriver.cuLaunchKernel(function,
                numBlocks, 1, 1, // Grid dimension
                blockSize, 1, 1, // Block dimension
                0, null, // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
        );
        JCudaDriver.cuCtxSynchronize();

        // Copy the output data from device to host
        float[] output_real = new float[totalSamples];
        float[] output_imag = new float[totalSamples];
        JCudaDriver.cuMemcpyDtoH(Pointer.to(output_real), d_output_real, totalSamples * Sizeof.FLOAT);
        JCudaDriver.cuMemcpyDtoH(Pointer.to(output_imag), d_output_imag, totalSamples * Sizeof.FLOAT);

        // Process the output (average amplitude, etc.)
        // You can implement your averaging logic here...

        // Clean up
        JCudaDriver.cuMemFree(d_input_real);
        JCudaDriver.cuMemFree(d_input_imag);
        JCudaDriver.cuMemFree(d_output_real);
        JCudaDriver.cuMemFree(d_output_imag);
        JCudaDriver.cuCtxDestroy(context);

        System.out.println("Execution Done. Outpu:");
        System.out.println("nReal: " + output_real.length + " | nImag: " + output_imag.length);
        float[] results = complexAbs(output_real, output_imag);
        for (float f : results){
            System.out.println(f);
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
