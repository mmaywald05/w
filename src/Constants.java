public class Constants {
    // Aus DeviceQuery.java

    static int THREADS_PER_BLOCK = 1024;
    static dim3 BLOCK_DIM = new dim3(1024,1024,64);
    static dim3 GRID_DIM =  new dim3(2147483647, 65535, 65535);
    static int SHARED_MEMORY_PER_BLOCK = 49152;
    static int CONSTANT_MEMORY = 65536;
}
class dim3{
    int x,y,z;
    public dim3(int x, int y, int z){
        this.x=x;
        this.y=y;
        this.z=z;
    }
    public int x(){
        return this.x;
    }
    public int y(){
        return this.y;
    }
    public int z(){
        return this.z;
    }

}
