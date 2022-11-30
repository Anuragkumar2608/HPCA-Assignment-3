#include <x86intrin.h>
// Optimize this function
//
//void singleThread(int N, int *matA, int *matB, int *output)
//{
//  assert( N>=4 and N == ( N &~ (N-1)));
//  for(int rowA = 0; rowA < N; rowA +=2) {
//    for(int colB = 0; colB < N; colB += 2){
//      int sum = 0;
//      for(int iter = 0; iter < N; iter++) 
//      {
//        sum += matA[rowA * N + iter] * matB[iter * N + colB];
//        sum += matA[(rowA+1) * N + iter] * matB[iter * N + colB];
//        sum += matA[rowA * N + iter] * matB[iter * N + (colB+1)];
//        sum += matA[(rowA+1) * N + iter] * matB[iter * N + (colB+1)];
//      }
//
//      // compute output indices
//      int rowC = rowA>>1;
//      int colC = colB>>1;
//      int indexC = rowC * (N>>1) + colC;
//      output[indexC] = sum;
//    }
//  }
//}

//void singleThread(int N, int *matA, int *matB, int *output)
//{
//  assert( N>=4 and N == ( N &~ (N-1)));
//  
//        int i, j, k;
//        for (j = 0; j < N; j+=2)
//        {   
//            int col[N];
//            int col2[N];
//            for (k=0; k<N; k++){
//                col[k] = matA[(k*N) + j];
//                col2[k] = matA[(k*N) + j+1];        
//            } 
//            for(i=0; i<N; i+=2){
//                int sum = 0;
//                for (k=0; k<N; k++){
//                    sum += mat1[(i*N)+k] * col[k];
//                    sum += mat1[(i*N)+k] * col2[k]; 
//                    sum += mat1[(i*(N+1))+k] * col[k];
//                    sum += mat1[(i*(N+1))+k] * col2[k];     
//                }
//            int r = i>>1;
//            int s = j>>1;
//            result[(r*N)+s] = sum; 
//    //        printf("%d",result[i][j]); 
//            }  
//        }
//   
//  for(int rowA = 0; rowA < N; rowA +=2) {
//    for(int colB = 0; colB < N; colB += 2){
//      int sum = 0;
//      for(int iter = 0; iter < N; iter+=8) 
//      {
//        __m256i vec1 = _mm256_mullo_epi32(_mm256_loadu_si256((__m256i*)&matA[rowA * N + iter]), _mm256_loadu_si256((__m256i*)&matB[iter * N + colB]) ) ;
//        sum += (vec1[0] + vec1[1] + vec1[2] + vec1[3] + vec1[4] + vec1[5] + vec1[6] + vec1[7]);
//        __m256i vec2 = _mm256_mullo_epi32(_mm256_loadu_si256((__m256i*)&matA[(rowA+1) * N + iter]), _mm256_loadu_si256((__m256i*)&matB[iter * N + colB]) ) ;
//        vec2 = _mm256_add_epi32(vec2,vec1);
//        sum += (vec2[0] + vec2[1] + vec2[2] + vec2[3] + vec2[4] + vec2[5] + vec2[6] + vec2[7]);
//        __m256i vec3 = _mm256_mullo_epi32(_mm256_loadu_si256((__m256i*)&matA[rowA * N + iter]), _mm256_loadu_si256((__m256i*)&matB[(iter+1) * N + colB]) ) ;
//        vec3 = _mm256_add_epi32(vec3,vec2);
//        sum += (vec3[0] + vec3[1] + vec3[2] + vec3[3] + vec3[4] + vec3[5] + vec3[6] + vec3[7]);
//        __m256i vec4 = _mm256_mullo_epi32(_mm256_loadu_si256((__m256i*)&matA[(rowA+1) * N + iter]), _mm256_loadu_si256((__m256i*)&matB[(iter+1) * N + colB]) ) ;
//        vec4 = _mm256_add_epi32(vec4,vec3);
//        int d[8];
//        _mm256_storeu_si256((__m256i*)&d, vec4);
//        sum += (d[0] + d[1] + d[2] + d[3] + d[4] + d[5] + d[6] + d[7]);
//        for(int i=0; i<8; i++){
//            printf("%d\t",d[i]);
//        }
//        printf("\n");
//
//      }
//
//      // compute output indices
//      int rowC = rowA>>1;
//      int colC = colB>>1;
//      int indexC = rowC * (N>>1) + colC;
//      output[indexC] = sum;
//    }
//  }
//}

void singleThread(int N, int *mat1, int *mat2, int *result)
{
  int i, j, k;
        for (j = 0; j < N; j+=2)
        {   
            int col[N];
            int col2[N];
            for (k=0; k<N; k++){
                col[k] = mat2[(k*N) + j];
                col2[k] = mat2[(k*N) + j+1];       
            } 
            for(i=0; i<N; i+=2){
                int sum = 0;
                for (k=0; k<N; k+=8){
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
            result[(r*(N>>1))+s] = sum; 
    //        printf("%d",result[i][j]); 
        }  
    }
}
