#include <stdlib.h>
#include <stdio.h>

__global__ void write(int *arr)
{
    int total = gridDim.x * blockDim.x;
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    arr[index] = total;
}

__global__ void count(int *k) {
    int total = gridDim.x * blockDim.x;
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    for (int i = 0; i < total; i++) {
        __syncthreads();
        if (i == idx)
            *k += 1;
    }
}

int main(int argc, char** argv)
{
    int k = 0;
    int *d_k;
    cudaMalloc((void**) &d_k, sizeof(int));
    cudaMemcpy(d_k, &k, sizeof(int), cudaMemcpyHostToDevice);

    count<<<2,6>>>(d_k);

    cudaDeviceSynchronize();
    cudaMemcpy(&k, d_k, sizeof(int), cudaMemcpyDeviceToHost);

    printf("count: %d\n", k);
    cudaFree(d_k);

    // int n = 16;
    // size_t size = n * sizeof(int);

    // int *arr = (int*) malloc(size);
    // int *d_arr;
    // cudaMalloc((void**) &d_arr, size);

    // cudaMemcpy(d_arr, arr, size, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // write<<<4,4>>>(d_arr); 

    // cudaMemcpy(arr, d_arr, size, cudaMemcpyDeviceToHost);

    // for (int i = 0; i < n; i++) {
    //     printf("%d\n", arr[i]);
    // }

    // free(arr);
    // cudaFree(d_arr);

    return 0;
}
