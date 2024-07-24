import java.util.concurrent.ExecutionException;


public class main {


    public static void main(String[] args) throws ExecutionException, InterruptedException {
        String filePath = "short210.wav";
        int blockSize = 512;
        int offset = 64;
        double threshold = (0.0d);
        int cores = Runtime.getRuntime().availableProcessors();

        System.out.println("Initializing sequential execution");
        long start_seq = System.currentTimeMillis();
        FFTFactory.DFT_SEQ(filePath, blockSize, offset, threshold);
        long end_seq = System.currentTimeMillis();
        long seqTime = end_seq - start_seq;




        System.out.println("Initializing parallel execution on cpu \n"
                +   "Cores: " + cores
        );
        long start_cpu = System.currentTimeMillis();
        FFTFactory.DFT_PAR(filePath, blockSize, offset, threshold);
        long end_cpu = System.currentTimeMillis();
        long cpuTime = end_cpu- start_cpu;

        System.out.println("Sequential Execution took " + seqTime +"ms");
        System.out.println("Parallel CPU Execution took " + cpuTime +"ms");


        double speedup_seq_cpu = (double) seqTime/cpuTime;
        System.out.println("Cores: "  +cores + " | Speedup SEQ->PAR_CPU: " + speedup_seq_cpu);

        //WavFileFactory.createSilentWav("chrissi", 600, 110);
        System.exit(0);
    }
}