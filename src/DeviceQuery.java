import jcuda.driver.CUdevice;
import jcuda.driver.CUdevprop;
import jcuda.driver.JCudaDriver;

import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuDeviceGetCount;
import static jcuda.driver.JCudaDriver.cuDeviceGetName;
import static jcuda.driver.JCudaDriver.cuDeviceGetProperties;
import static jcuda.driver.JCudaDriver.cuInit;

public class DeviceQuery {
    public static void main(String[] args) {
        // Initialize JCuda
        JCudaDriver.setExceptionsEnabled(true);
        cuInit(0);

        // Obtain the number of devices
        int deviceCountArray[] = {0};
        cuDeviceGetCount(deviceCountArray);
        int deviceCount = deviceCountArray[0];
        System.out.println("Number of devices: " + deviceCount);

        // Iterate over all devices
        for (int i = 0; i < deviceCount; i++) {
            CUdevice device = new CUdevice();
            cuDeviceGet(device, i);

            // Obtain the device name
            byte deviceName[] = new byte[256];
            cuDeviceGetName(deviceName, deviceName.length, device);
            String deviceNameString = new String(deviceName).trim();
            System.out.println("Device " + i + ": " + deviceNameString);

            // Obtain the device properties
            CUdevprop deviceProperties = new CUdevprop();
            cuDeviceGetProperties(deviceProperties, device);

            // Print the device properties
            System.out.println("  Max threads per block: " + deviceProperties.maxThreadsPerBlock);
            System.out.println("  Max block dimensions: " +
                    deviceProperties.maxThreadsDim[0] + " x " +
                    deviceProperties.maxThreadsDim[1] + " x " +
                    deviceProperties.maxThreadsDim[2]);
            System.out.println("  Max grid dimensions: " +
                    deviceProperties.maxGridSize[0] + " x " +
                    deviceProperties.maxGridSize[1] + " x " +
                    deviceProperties.maxGridSize[2]);
            System.out.println("  Shared memory per block: " + deviceProperties.sharedMemPerBlock);
            System.out.println("  Total constant memory: " + deviceProperties.totalConstantMemory);
            System.out.println("  Warp size: " + deviceProperties.SIMDWidth);
            System.out.println("  Max pitch: " + deviceProperties.memPitch);
            System.out.println("  Registers per block: " + deviceProperties.regsPerBlock);
            System.out.println("  Clock rate: " + deviceProperties.clockRate);
            System.out.println("  Texture alignment: " + deviceProperties.textureAlign);
        }
    }
}
