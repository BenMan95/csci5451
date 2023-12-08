#include <stdlib.h>
#include <stdio.h>

__global__ void add(int *a, int *b, int *c)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    c[index] = a[index] + b[index];
}

int main(int argc, char** argv)
{
    int n = 5;
    size_t size = n * sizeof(int);

    int *a = (int*) malloc(size);
    int *b = (int*) malloc(size);
    int *c = (int*) malloc(size);

    int *d_a, *d_b, *d_c;
    cudaMalloc((void**) &d_a, size);
    cudaMalloc((void**) &d_b, size);
    cudaMalloc((void**) &d_c, size);

    for (int i = 0; i < n; i++) {
        a[i] = i;
        b[i] = i+1;
    }

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    add<<<1,n>>>(d_a, d_b, d_c); 

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    free(a);
    free(b);
    free(c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}
