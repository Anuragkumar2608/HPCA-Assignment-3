// Create other necessary functions here
#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>

// #define N 4096

__global__ void matrixMul(const int *a, const int *b, int *c, int N){
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int rowA=row;
  int colB=col;

if (((row %2)==0) && (col%2==0) && (row<N) && (col+1<N))
{
 int sum=0;
    for(int iter = 0; iter < N; iter++) 
     {
       sum += a[rowA * N + iter] * b[iter * N + colB];
       sum += a[(rowA+1) * N + iter] * b[iter * N + colB];
       sum += a[rowA * N + iter] * b[iter * N + (colB+1)];
       sum += a[(rowA+1) * N + iter] * b[iter * N + (colB+1)];
     }
     int rowC = rowA>>1;
     int colC = colB>>1;
     int indexC = rowC * (N>>1) + colC;
     c[indexC] = sum;
}
}


// Fill in this function
void gpuThread(int N, int *matA, int *matB, int *output)
{
  

  int *d_a, *d_b, *d_c;
  cudaMalloc((void**)&d_a, N*N*sizeof(int));
  cudaMalloc((void**)&d_b, N*N*sizeof(int));
  cudaMalloc((void**)&d_c, N/2*N/2*sizeof(int));

  // Copy data to the device
  cudaMemcpy(d_a, matA, N*N*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, matB, N*N*sizeof(int), cudaMemcpyHostToDevice);
//  cudaMemcpy(d_c, output, N/2*N/2*sizeof(int), cudaMemcpyHostToDevice);


  // Threads per CTA dimension
  int THREADS = 1;

  // Blocks per grid dimension (assumes THREADS divides N evenly)
  int BLOCKS = N/THREADS ;

  // // Use dim3 structs for block  and grid dimensions
   dim3 threads(THREADS, THREADS);
   dim3 blocks(BLOCKS, BLOCKS);

  // Launch kernel
  matrixMul<<<blocks, threads>>>(d_a,d_b,d_c,N);

  // Copy back to the host
  cudaMemcpy(output, d_c, N/2*N/2*sizeof(int),cudaMemcpyDeviceToHost);


//  printf("COMPLETED SUCCESSFULLY\n");

  // Free memory on device
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);


}
