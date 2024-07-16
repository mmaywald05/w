extern "C"
__global__ void fftKernel(float2* data, int n, int step) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        int j = index / step * step * 2 + index % step;
        int k = j + step;

        float2 u = data[j];
        float2 t = data[k];
        float angle = -2.0f * 3.14159265358979323846f * (index % step) / (2.0f * step);
        float2 w = make_float2(cosf(angle), sinf(angle));

        float2 temp;
        temp.x = w.x * t.x - w.y * t.y;
        temp.y = w.x * t.y + w.y * t.x;

        data[j] = make_float2(u.x + temp.x, u.y + temp.y);
        data[k] = make_float2(u.x - temp.x, u.y - temp.y);
    }
}
