#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <pthread.h>


// Create other necessary functions here

int N, num_threads = 8;
int *mat1, *mat2, *matrix3;

void *worker( void *arg )
{
  int tid, portion_size, row_start, row_end;
  double sum;
  
  tid = *(int *)(arg); 
  portion_size = N / num_threads;
  row_start = tid * portion_size;
  row_end = (tid+1) * portion_size;

//  for (int i = row_start; i < row_end; i+=2) {
//    for (int j = 0; j < N; j+=2) {
//      sum = 0; // hold value of a cell
//      for (int k = 0; k < N; ++k) { 
//	    sum += matrix1[i * N + k] * matrix2[k * N + j];
//        sum += matrix1[(i+1) * N + k] * matrix2[k * N + j];
//        sum += matrix1[i * N + k] * matrix2[k * N + (j+1)];
//        sum += matrix1[(i+1) * N + k] * matrix2[k * N + (j+1)];
//    }
//      int rowC = i>>1;
//      int colC = j>>1;
//      int indexC = rowC * (N>>1) + colC;
//      matrix3[indexC] = sum;
//  }
//  }

    for (int j = row_start; j < row_end; j+=2)
        {   
            int col[N];
            int col2[N];
            for (int k=0; k<N; k++){
                col[k] = mat2[(k*N) + j];
                col2[k] = mat2[(k*N) + j+1];       
            } 
            for(int i=0; i<N; i+=2){
                int sum = 0;
                for (int k=0; k<N; k+=8){
//                    sum += mat1[(i*N)+k] * col[k];
//                    sum += mat1[(i*N)+k] * col2[k]; 
//                    sum += mat1[(i+1)*(N)+k] * col[k];
//                    sum += mat1[(i+1)*(N)+k] * col2[k];   

                        __m256i vec1 = _mm256_mullo_epi32(_mm256_loadu_si256((__m256i*)&mat1[i * N + k]), _mm256_loadu_si256((__m256i*)&col[k]) ) ;
                //        sum += (vec1[0] + vec1[1] + vec1[2] + vec1[3] + vec1[4] + vec1[5] + vec1[6] + vec1[7]);
                        __m256i vec2 = _mm256_mullo_epi32(_mm256_loadu_si256((__m256i*)&mat1[i * N + k]), _mm256_loadu_si256((__m256i*)&col2[k]) ) ;
                        vec2 = _mm256_add_epi32(vec2,vec1);
                //        sum += (vec2[0] + vec2[1] + vec2[2] + vec2[3] + vec2[4] + vec2[5] + vec2[6] + vec2[7]);
                        __m256i vec3 = _mm256_mullo_epi32(_mm256_loadu_si256((__m256i*)&mat1[(i+1)*(N)+k]), _mm256_loadu_si256((__m256i*)&col[k]) ) ;
                        vec3 = _mm256_add_epi32(vec3,vec2);
                //        sum += (vec3[0] + vec3[1] + vec3[2] + vec3[3] + vec3[4] + vec3[5] + vec3[6] + vec3[7]);
                        __m256i vec4 = _mm256_mullo_epi32(_mm256_loadu_si256((__m256i*)&mat1[(i+1)*(N)+k]), _mm256_loadu_si256((__m256i*)&col2[k]) ) ;
                        vec4 = _mm256_add_epi32(vec4,vec3);
                        int d[8];
                        _mm256_storeu_si256((__m256i*)&d, vec4);
                        sum += (d[0] + d[1] + d[2] + d[3] + d[4] + d[5] + d[6] + d[7]);
  
                }
            int r = i>>1;
            int s = j>>1;
            matrix3[(r*(N>>1))+s] = sum; 
    //        printf("%d",result[i][j]); 
        }  
    }

}




void multiThread(int n, int *matA, int *matB, int *output)
{
  N = n;
  mat1 = &matA[0];
  mat2 = &matB[0];
  matrix3 = &output[0];
  pthread_t * threads;

  threads = (pthread_t *) malloc( num_threads * sizeof(pthread_t) );

  for (int i = 0; i < num_threads; ++i ) {
    int *tid;
    tid = (int *) malloc( sizeof(int) );
    *tid = i;
    pthread_create( &threads[i], NULL, worker, (void *)tid );
  }

  for (int i = 0; i < num_threads; ++i ) {
    pthread_join( threads[i], NULL );
  }
    
}

