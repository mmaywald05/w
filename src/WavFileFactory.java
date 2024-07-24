import javax.sound.sampled.*;
import java.io.*;

public class WavFileFactory {

    public static void createSilentWav(String filename, int durationSec, double frequency) {
        float sampleRate = 44100.0f; // 44.1kHz
        int sampleSizeInBits = 16; // 2 bytes per frame (16-bit PCM)
        int channels = 1; // Mono
        boolean signed = true;
        boolean bigEndian = false;

        // Create audio format
        AudioFormat format = new AudioFormat(sampleRate, sampleSizeInBits, channels, signed, bigEndian);

        // Calculate the total number of frames
        int numFrames = (int) (sampleRate * durationSec);
        byte[] audioData = new byte[numFrames * 2]; // 2 bytes per frame

        // Frequency of the A440 tone

        double amplitude = 32767.0; // Max amplitude for 16-bit audio
        double twoPiF = 2 * Math.PI * frequency;

        // Generate the sine wave data
        for (int i = 0; i < numFrames; i++) {
            double time = i / sampleRate;
            short value = (short) (amplitude * Math.sin(twoPiF * time));
            audioData[2 * i] = (byte) (value & 0xff);
            audioData[2 * i + 1] = (byte) ((value >> 8) & 0xff);
        }

        // Create the audio input stream
        ByteArrayInputStream bais = new ByteArrayInputStream(audioData);
        AudioInputStream ais = new AudioInputStream(bais, format, numFrames);

        // Write the audio data to a WAV file
        try {
            File file = new File(filename+".wav");
            if (AudioSystem.write(ais, AudioFileFormat.Type.WAVE, file) == -1) {
                throw new IOException("Problems writing to file");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    public static void create(String filename, int durationSec, double frequency) {
        float sampleRate = 44100.0f; // 44.1kHz
        int sampleSizeInBits = 16; // 2 bytes per frame (16-bit PCM)
        int channels = 1; // Mono
        boolean signed = true;
        boolean bigEndian = false;

        // Create audio format
        AudioFormat format = new AudioFormat(sampleRate, sampleSizeInBits, channels, signed, bigEndian);

        // Calculate the total number of frames
        int numFrames = (int) (sampleRate * durationSec);
        byte[] audioData = new byte[numFrames * 2]; // 2 bytes per frame

        // Frequency of the A440 tone

        double amplitude = 32767.0; // Max amplitude for 16-bit audio
        double twoPiF = 2 * Math.PI * frequency;

        // Generate the sine wave data
        for (int i = 0; i < numFrames; i++) {
            double time = i / sampleRate;
            short value = (short) (amplitude * Math.sin(twoPiF * time));
            audioData[2 * i] = (byte) (value & 0xff);
            audioData[2 * i + 1] = (byte) ((value >> 8) & 0xff);
        }

        // Create the audio input stream
        ByteArrayInputStream bais = new ByteArrayInputStream(audioData);
        AudioInputStream ais = new AudioInputStream(bais, format, numFrames);

        // Write the audio data to a WAV file
        try {
            File file = new File(filename+".wav");
            if (AudioSystem.write(ais, AudioFileFormat.Type.WAVE, file) == -1) {
                throw new IOException("Problems writing to file");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static float[] readWavFile(String filePath) throws IOException, UnsupportedAudioFileException, UnsupportedAudioFileException {
        File file = new File(filePath);
        AudioInputStream audioInputStream = AudioSystem.getAudioInputStream(file);
        AudioFormat format = audioInputStream.getFormat();

        if (format.getEncoding() != AudioFormat.Encoding.PCM_SIGNED) {
            throw new UnsupportedAudioFileException("Only PCM_SIGNED encoding is supported.");
        }

        int numChannels = format.getChannels();
        int bytesPerSample = format.getSampleSizeInBits() / 8;
        int frameSize = format.getFrameSize();
        int bufferSize = 1024 * frameSize; // arbitrary buffer size

        byte[] buffer = new byte[bufferSize];
        int availableBytes = (int) (file.length() - (file.length() % frameSize));
        float[] samples = new float[availableBytes / bytesPerSample];

        int bytesRead;
        int sampleIndex = 0;
        float maxAbsValue = 0;

        while ((bytesRead = audioInputStream.read(buffer)) != -1) {
            for (int i = 0; i < bytesRead; i += bytesPerSample) {
                int sample = 0;
                for (int j = 0; j < bytesPerSample; j++) {
                    sample |= (buffer[i + j] & 0xFF) << (j * 8);
                }
                if (bytesPerSample == 2) {
                    sample = (short) sample; // Convert to signed 16-bit
                }
                float sampleFloat = sample / (float) (Math.pow(2, format.getSampleSizeInBits() - 1));
                samples[sampleIndex++] = sampleFloat;
                maxAbsValue = Math.max(maxAbsValue, Math.abs(sampleFloat));
            }
        }

        audioInputStream.close();

        // Normalize samples to [-1, 1]
        if (maxAbsValue > 0) {
            for (int i = 0; i < samples.length; i++) {
                samples[i] /= maxAbsValue;
            }
        }

        return samples;
    }


    public static void main(String[] args) {
        WavFileFactory.createSilentWav("short210", 10, 210);
    }

    public static void writeFloatArrayToFile(float[] floatArray, String fileName) throws IOException {
        // Use try-with-resources to ensure that the BufferedWriter is closed properly
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(fileName))) {
            for (float value : floatArray) {
                writer.write(Float.toString(value));
                writer.newLine();
            }
        }
    }

}

