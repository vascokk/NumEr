#ifndef numer_CUBLAS_WRAPPERS
#define numer_CUBLAS_WRAPPERS

#include "cuda.h"
#include "cublas_v2.h"
#include <stdio.h>

//template<> void cublasGemm<int> (int param){

template <typename T>
class BlasWrapper{
public:
  cublasStatus_t cublasGemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const T *alpha, const T *A, int lda, const T *B, int ldb, const T *beta, T *C, int ldc);

  cublasStatus_t cublasGemv(cublasHandle_t handle, cublasOperation_t trans,
                             int m, int n,
                             const T           *alpha,
                             const T           *A, int lda,
                             const T           *x, int incx,
                             const T           *beta,
                             T           *y, int incy);


  cublasStatus_t cublasGeam(cublasHandle_t handle,
                            cublasOperation_t transa, cublasOperation_t transb,
                            int m, int n,
                            const T           *alpha,
                            const T           *A, int lda,
                            const T           *beta,
                            const T           *B, int ldb,
                            T           *C, int ldc);

};
#endif