#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>



#define N 10000
#define S N*sizeof(int)


// kerenl, the parallelized proccess
__global__ void VAdding(int *a, int *b, int *c){
    int i = threadIdx.x;
    int j = blockIdx.x;
    int globalindex = blockDim.x * j + i;
    c[globalindex] = a[globalindex] + b[globalindex];

}

int main(void)
{
   
   int a[N];
   int b[N];
   int c[N];

   int *Ca,*Cb,*Cc;
   for(int i = 0; i <N;i++){
     a[i]= -i;
     b[i]= i*i;
   } 

   cudaMalloc((void**)&Ca,S); 
   cudaMalloc((void**)&Cb,S); 
   cudaMalloc((void**)&Cc,S); 


   cudaMemcpy(Ca,a, S,cudaMemcpyHostToDevice);
   cudaMemcpy(Cb,b, S,cudaMemcpyHostToDevice); 

    VAdding<<<100,100>>>(Ca,Cb,Cc);

    cudaMemcpy(c,Cc, S,cudaMemcpyDeviceToHost);


    for(int i = 0; i<N; i+=100){
        
        printf("%d + %d = %d \n",a[i],b[i],c[i]);
    }

    
    cudaFree(Ca);
    cudaFree(Cb);
    cudaFree(Cc);

    return 0;
}


