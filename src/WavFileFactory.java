import javax.sound.sampled.AudioFileFormat;
import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;

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
}

