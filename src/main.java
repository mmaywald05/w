import javax.sound.sampled.*;
import java.io.File;
import java.io.IOException;
import java.sql.SQLOutput;


public class main {


    public static void main(String[] args) {
        String filePath = "Chrissi.wav";
        int blockSize = 1024;
        int offset = 64;
        double threshold = (0.0d);


        long start_seq = System.currentTimeMillis();
        FFTFactory.FFT_SEQ(filePath, blockSize, offset, threshold);
        long end_seq = System.currentTimeMillis();

        long seqTime = end_seq - start_seq;




        System.out.println("Initializing parallel execution");
        long start_cpu = System.currentTimeMillis();
        FFTFactory.FFT_PAR_CPU(filePath, blockSize, offset, threshold);
        long end_cpu = System.currentTimeMillis();
        long cpuTime = end_cpu- start_cpu;

        System.out.println("Sequential Execution took " + seqTime +"ms");

        System.out.println("Parallel CPU Execution took " + cpuTime +"ms");


/*
        long start_it = System.currentTimeMillis();
        FFTFactory.CK_IterativeFFT(filePath, blockSize, offset, threshold);
        long end_it = System.currentTimeMillis();
        long itTime = end_it - start_it;

        System.out.println("Iterative Cooley-Tukey took "+ itTime+ "ms.");


 */


        int cores = Runtime.getRuntime().availableProcessors();
        //double speedup_seq_cpu = (double) seqTime/cpuTime;
        //System.out.println("Cores: "  +cores + " | Speedup SEQ->PAR_CPU: " + speedup_seq_cpu);



        //WavFileFactory.createSilentWav("chrissi", 600, 110);
        System.exit(0);
    }
}