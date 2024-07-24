import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.UnsupportedAudioFileException;
import java.awt.event.ComponentListener;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class FFTFactory {
    public static void FFT_SEQ(String filePath, int blockSize, int offset, double threshold) {

        try {
            AudioInputStream audioInputStream = AudioSystem.getAudioInputStream(new File(filePath));
            AudioFormat format = audioInputStream.getFormat();
            int bytesPerFrame = format.getFrameSize();
            System.out.println("Bit Depth: " + (bytesPerFrame*8));
            int sampleRate = (int) format.getSampleRate();
            System.out.println("samplerate: " + sampleRate);

            byte[] audioBytes = readAllBytes(audioInputStream);
            int numSamples = audioBytes.length / bytesPerFrame;
            double[] samples = new double[numSamples];



            for (int i = 0; i < numSamples; i++) {
                int sampleStart = i * bytesPerFrame;
                double sample = 0;
                for (int byteIndex = 0; byteIndex < bytesPerFrame; byteIndex++) {
                    sample += ((int) audioBytes[sampleStart + byteIndex] << (8 * byteIndex));
                }
                samples[i] = sample;
            }


            System.out.println("samples size: " + samples.length);

            // Alle Samples der Datei als Double in samples[]. ||  1 323 000 Samples in Total
            // NumBlocKs = 42 312
            int numBlocks = (numSamples - blockSize) / offset + 1;



            double[] amplitudeSums = new double[blockSize / 2];

            for (int blockIndex = 0; blockIndex < numBlocks; blockIndex++) {
                int blockStart = blockIndex * offset;
                double[] block = new double[blockSize];
                System.arraycopy(samples, blockStart, block, 0, blockSize);

                double[] fftResult = fft(block);

                for (int i = 0; i < blockSize / 2; i++) {
                    double real = fftResult[2 * i];
                    double imag = fftResult[2 * i + 1];
                    double amplitude = Math.sqrt(real * real + imag * imag);
                    amplitudeSums[i] += amplitude;
                }
            }



            System.out.println("Frequenz (Hz)\tAmplitudenmittelwert");
            for (int i = 0; i < amplitudeSums.length; i++) {
                double averageAmplitude = amplitudeSums[i] / numBlocks;
                if (averageAmplitude > threshold) {
                    double frequency = (double) i * sampleRate / blockSize;
                    System.out.printf("%.2f\t\t%.5f%n", frequency, averageAmplitude);
                }
            }



        } catch (UnsupportedAudioFileException | IOException e) {
            System.out.println("Joer`?");
            e.printStackTrace();
        }
    }


    public static void FFT_PAR_CPU(String filePath, int blockSize, int offset, double threshold) {
      /*
        if (blockSize < 64 || blockSize > 512) {
            throw new IllegalArgumentException("Blockgröße muss zwischen 64 und 512 liegen.");
        }
        if (offset < 1 || offset > blockSize) {
            throw new IllegalArgumentException("Versatz muss zwischen 1 und Blockgröße liegen.");
        }

       */

        try {
            AudioInputStream audioInputStream = AudioSystem.getAudioInputStream(new File(filePath));
            AudioFormat format = audioInputStream.getFormat();
            int bytesPerFrame = format.getFrameSize();
            System.out.println("bytesPerFrame: " + bytesPerFrame);
            int sampleRate = (int) format.getSampleRate();
            System.out.println("samplerate: " + sampleRate);

            byte[] audioBytes = readAllBytes(audioInputStream);
            int numSamples = audioBytes.length / bytesPerFrame;
            double[] samples = new double[numSamples];

            for (int i = 0; i < numSamples; i++) {
                int sampleStart = i * bytesPerFrame;
                double sample = 0;
                for (int byteIndex = 0; byteIndex < bytesPerFrame; byteIndex++) {
                    sample += ((int) audioBytes[sampleStart + byteIndex] << (8 * byteIndex));
                }
                samples[i] = sample;
            }

            int numBlocks = (numSamples - blockSize) / offset + 1;
            double[] amplitudeSums = new double[blockSize / 2];

            // Initialize the ExecutorService with the number of available cores
            int numCores = Runtime.getRuntime().availableProcessors();
            ExecutorService executor = Executors.newFixedThreadPool(numCores);
            List<Future<double[]>> futures = new ArrayList<>();

            // Submit tasks for parallel processing
            for (int blockIndex = 0; blockIndex < numBlocks; blockIndex++) {
                final int startIndex = blockIndex * offset;
                futures.add(executor.submit(() -> {
                    double[] block = new double[blockSize];
                    System.arraycopy(samples, startIndex, block, 0, blockSize);
                    return fft(block);
                }));
            }

            // Combine results
            for (Future<double[]> future : futures) {
                double[] fftResult = future.get();
                for (int i = 0; i < blockSize / 2; i++) {
                    double real = fftResult[2 * i];
                    double imag = fftResult[2 * i + 1];
                    double amplitude = Math.sqrt(real * real + imag * imag);
                    amplitudeSums[i] += amplitude;
                }
            }


            System.out.println("Frequenz (Hz)\tAmplitudenmittelwert");
            for (int i = 0; i < amplitudeSums.length; i++) {
                double averageAmplitude = amplitudeSums[i] / numBlocks;
                if (averageAmplitude > threshold) {
                    double frequency = (double) i * sampleRate / blockSize;
                    System.out.printf("%.2f\t\t%.5f%n", frequency, averageAmplitude);
                }
            }



        } catch (UnsupportedAudioFileException | IOException | ExecutionException | InterruptedException e) {
            System.out.println("fail.");
            e.printStackTrace();
        }
    }


    public static Complex[] loadSamplesAsComplex (String filePath){

        try {
            float[] samples = WavFileFactory.readWavFile(filePath);


            Complex[] complex = new Complex[samples.length];
            for (int i = 0; i < samples.length; i++) {
                complex[i] = new Complex(samples[i], 0);
            }

            return complex;

        } catch (UnsupportedAudioFileException | IOException e) {
            System.out.println("Error reading file`?");
            e.printStackTrace();
            return null;
        }
    }


    public static void CK_IterativeFFT(String filePath, int blockSize, int offset, double threshold){

        try {
            AudioInputStream audioInputStream = AudioSystem.getAudioInputStream(new File(filePath));
            AudioFormat format = audioInputStream.getFormat();
            int bytesPerFrame = format.getFrameSize();
            System.out.println("Bit Depth: " + (bytesPerFrame*8));
            int sampleRate = (int) format.getSampleRate();
            System.out.println("samplerate: " + sampleRate);

            byte[] audioBytes = readAllBytes(audioInputStream);
            int numSamples = audioBytes.length / bytesPerFrame;
            double[] samples = new double[numSamples];



            for (int i = 0; i < numSamples; i++) {
                int sampleStart = i * bytesPerFrame;
                double sample = 0;
                for (int byteIndex = 0; byteIndex < bytesPerFrame; byteIndex++) {
                    sample += ((int) audioBytes[sampleStart + byteIndex] << (8 * byteIndex));
                }
                samples[i] = sample;
            }


            System.out.println("samples size: " + samples.length);

            // Alle Samples der Datei als Double in samples[]. ||  1 323 000 Samples in Total
            // NumBlocKs = 42 312
            int numBlocks = (numSamples - blockSize) / offset + 1;



            double[] amplitudeSums = new double[blockSize / 2];

            for (int blockIndex = 0; blockIndex < numBlocks; blockIndex++) {
                int blockStart = blockIndex * offset;
                double[] real_arr = new double[blockSize];
                System.arraycopy(samples, blockStart, real_arr, 0, blockSize);

                double[] imaginary_arr = new double[blockSize];

                CoolTuk_iterative(real_arr, imaginary_arr);

                for (int i = 0; i < blockSize / 2; i++) {
                    double amplitude = Math.sqrt(real_arr[i]*real_arr[i] + imaginary_arr[i] * imaginary_arr[i]);
                    amplitudeSums[i] += amplitude;
                }
            }



            System.out.println("Frequenz (Hz)\tAmplitudenmittelwert");
            for (int i = 0; i < amplitudeSums.length; i++) {
                double averageAmplitude = amplitudeSums[i] / numBlocks;
                if (averageAmplitude > threshold) {
                    double frequency = (double) i * sampleRate / blockSize;
                    System.out.printf("%.2f\t\t%.5f%n", frequency, averageAmplitude);
                }
            }

        } catch (UnsupportedAudioFileException | IOException e) {
            System.out.println("Joer`?");
            e.printStackTrace();
        }
    }

    public static byte[] readAllBytes(AudioInputStream audioInputStream) throws IOException {
        ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        byte[] data = new byte[1024];
        int bytesRead;

        while ((bytesRead = audioInputStream.read(data, 0, data.length)) != -1) {
            buffer.write(data, 0, bytesRead);
        }

        return buffer.toByteArray();
    }

    private static double[] fft(double[] input) {
        int n = input.length;
        if (n == 1) return new double[]{input[0]};

        if (Integer.highestOneBit(n) != n) {
            throw new IllegalArgumentException("Input array length must be a power of 2");
        }

        double[] output = new double[n * 2];
        for (int i = 0; i < n; i++) {
            output[2 * i] = input[i];
            output[2 * i + 1] = 0;
        }

        fftRecursive(output, n);
        return output;
    }


    public static void DFT_SEQ(String filePath, int blockSize, int shift, double threshold) {

        Complex[] input = loadSamplesAsComplex(filePath);
        int N = input.length;
        float[] avgMagnitudes = new float[blockSize];
        int numBlocks = (N - blockSize) / shift + 1 ;

        blockwise_DFT(input, avgMagnitudes, N, blockSize, shift, numBlocks);
        /*
        System.out.println("Starting Sequential DFT on CPU");
        for (int blockIndex = 0; blockIndex < numBlocks; blockIndex++){
            int blockStart = blockIndex * shift;
            Complex[] block = new Complex[blockSize];
            float[] blockMagnitudes = new float[blockSize];

            System.arraycopy(input, blockStart, block, 0, blockSize);
            DFT(block, blockMagnitudes, N, blockSize, shift, numBlocks);

            for (int i = 0; i <blockSize; i++) {
                avgMagnitudes[i] += blockMagnitudes[i] / numBlocks;
            }
        }


         */

        normalizeMagnitudes(avgMagnitudes);

        try {
            WavFileFactory.writeFloatArrayToFile(avgMagnitudes,"javamag.txt");
        } catch (IOException e) {
            e.printStackTrace();
        }
        for (int i = 0; i < avgMagnitudes.length; i++) {
            System.out.println("Bin: " + i + " Magnitude = " + (avgMagnitudes[i]));
        }
    }

    private static void DFT(Complex[] input, float[] magnitudes, int N, int k, int s, int numBlocks){
        for (int i= 0; i < input.length; i++) {
            Complex number = new Complex(0, 0);
            for (int j = 0; j < input.length; j++) {
                double angle = 2 * Math.PI * i * j / input.length;
                Complex w = new Complex((float) Math.cos(angle), (float) -Math.sin(angle));
                Complex product = Complex.multiply(input[j], w);
                number = Complex.add(number, product);
            }
            float mag = Complex.magnitude(number);
            magnitudes[i] += mag;
        }
    }

    private static void blockwise_DFT(Complex[] input, float[] magnitudes, int N, int k, int s, int numBlocks){

        for (int blockId = 0; blockId < numBlocks; blockId++) {
            int startIndex = blockId * s;
            int endIndex = startIndex + k;
            for (int i = startIndex; i < endIndex; i++) {
                Complex number = new Complex(0f, 0f);
                for (int j = startIndex; j < endIndex; j++) {
                    double angle = 2 * Math.PI * i * j / k;
                    Complex w = new Complex(Math.cos(angle), Math.cos(angle));
                    w = Complex.multiply(input[j], w);
                    number = Complex.add(number, w);
                }

                float mag = Complex.magnitude(number) / numBlocks;
                magnitudes[i-startIndex] += mag;
            }
        }
    }

    private static void fftRecursive(double[] x, int n) {
        if (n <= 1) return;

        double[] even = new double[n];
        double[] odd = new double[n];
        for (int i = 0; i < n / 2; i++) {
            even[2 * i] = x[2 * i * 2];
            even[2 * i + 1] = x[2 * i * 2 + 1];
            odd[2 * i] = x[2 * i * 2 + 2];
            odd[2 * i + 1] = x[2 * i * 2 + 3];
        }

        fftRecursive(even, n / 2);
        fftRecursive(odd, n / 2);

        for (int i = 0; i < n / 2; i++) {
            double theta = -2 * Math.PI * i / n;
            double cos = Math.cos(theta);
            double sin = Math.sin(theta);
            double real = cos * odd[2 * i] - sin * odd[2 * i + 1];
            double imag = sin * odd[2 * i] + cos * odd[2 * i + 1];

            x[2 * i] = even[2 * i] + real;
            x[2 * i + 1] = even[2 * i + 1] + imag;
            x[2 * (i + n / 2)] = even[2 * i] - real;
            x[2 * (i + n / 2) + 1] = even[2 * i + 1] - imag;
        }
    }

    private static void bitReverse(double[] a) {
        int n = a.length;
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

    // Cooley-Tukey FFT aber iterativ
    public static void CoolTuk_iterative(double[] x, double[] y) {
        // Assume n is a power of 2
        int n = x.length;
        bitReverse(x);

        // Compute FFT
        for (int len = 2; len <= n; len <<= 1) {
            double angle = 2 * Math.PI / len;
            double wlenR = Math.cos(angle);
            double wlenI = Math.sin(angle);
            for (int i = 0; i < n; i += len) {
                double wr = 1, wi = 0;
                for (int j = 0; j < len / 2; j++) {
                    double tempr = wr * x[i + j + len / 2] - wi * y[i + j + len / 2];
                    double tempi = wr * y[i + j + len / 2] + wi * x[i + j + len / 2];
                    double xr = x[i + j];
                    double xi = y[i + j];
                    x[i + j] = xr + tempr;
                    y[i + j] = xi + tempi;
                    x[i + j + len / 2] = xr - tempr;
                    y[i + j + len / 2] = xi - tempi;
                    double temp = wr;
                    wr = wr * wlenR - wi * wlenI;
                    wi = temp * wlenI + wi * wlenR;
                }
            }
        }
    }

    public static void normalizeMagnitudes(float[] magnitudes){
        float max = Float.MIN_VALUE;
        float min = Float.MAX_VALUE;

        for (int i = 0; i < magnitudes.length; i++) {
            if(magnitudes[i] < min)
                min = magnitudes[i];
            if(magnitudes[i] > max)
                max = magnitudes[i];
        }
        for (int i = 0; i < magnitudes.length; i++) {
            magnitudes[i] = (magnitudes[i]-min)/(max-min);
        }
    }

    public static class Complex {
        float x;
        float y;

        public Complex(float x, float y){
            this.x = x;
            this.y = y;
        }
        public Complex(double x, double y){
            this.x = (float) x;
            this.y = (float) y;
        }

        public float x(){ return this.x;}
        public float y(){return this.y;}

        public static Complex add(Complex a, Complex b){
            return new Complex(a.x + b.x, a.y + b.y);
        }
        public static Complex multiply(Complex a, Complex b){
            return new Complex(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
        }
        public static float magnitude(Complex a){
            return (float) Math.sqrt(a.x * a.x + a.y * a.y);
        }
    }
}
