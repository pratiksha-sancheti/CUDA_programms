#include<iostream>
#include<cstdlib>
#include<cmath>
#include<time.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 10000000
#define MAX_ERR 1e-6

using namespace std;

__global__ void vector_add(float *out, float *a, float *b, int n) {
   
   int i = threadIdx.x + blockIdx.x * blockDim.x;
   if(i<N){
    out[i]=a[i]+b[i];
   }
}

int main(){
    float *a, *b, *out,*cpu_out;
    float *d_a, *d_b, *d_out; 
    // Allocate host memory
    a   = (float*)malloc(sizeof(float) * N);
    b   = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);
    cpu_out = (float*)malloc(sizeof(float) * N);
    // Initialize host arrays
    for(int i = 0; i < N; i++){
        a[i] = i*1.0f;
        b[i] = i*1.0f;
    }

    // Allocate device memory 
    cudaMalloc((void**)&d_a, sizeof(float) * N);
    cudaMalloc((void**)&d_b, sizeof(float) * N);
    cudaMalloc((void**)&d_out, sizeof(float) * N);

    // Transfer data from host to device memory
    cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

    // Executing kernel 

    int block_size = 256;
    int grid_size = ((N + block_size) / block_size);
    vector_add<<<grid_size,block_size>>>(d_out, d_a, d_b, N);
    
    // Transfer data back to host memory
    cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

    clock_t t=clock();
    for(int i=0;i<N;i++){
        cpu_out[i] = a[i]+b[i];
    }
     t=clock()-t;
        cout<<"\nCPU Time Elapsed:  "<<((double)t)<<"\n"; 

    // Verification
    for(int i = 0; i < N; i++){
        assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
    }

    printf("PASSED\n");

	// for(int i=0;i<N;i++)
	// printf("%lf ",out[i]);

    // Deallocate device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    // Deallocate host memory
    free(a); 
    free(b); 
    free(out);
}