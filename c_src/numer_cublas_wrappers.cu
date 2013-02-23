#include "cuda.h"
#include "cublas_v2.h"
#include <stdio.h>
#include <iostream>
#include "numer_cublas_wrappers.h"

template<>
cublasStatus_t BlasWrapper<float>::cublasGemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, 
						   int m, int n, int k,
                           const float           *alpha,
                           const float           *A, int lda,
                           const float           *B, int ldb,
                           const float           *beta,
                           float           *C, int ldc){
  //std::cout<<"blas wrapper: float"<<std::endl;

	return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
};

template<>
cublasStatus_t BlasWrapper<double>::cublasGemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, 
						   int m, int n, int k,
                           const double           *alpha,
                           const double           *A, int lda,
                           const double           *B, int ldb,
                           const double           *beta,
                           double           *C, int ldc){

  //std::cout<<"blas wrapper: double"<<std::endl;

	return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
};

/*template<typename T>
cublasStatus_t BlasWrapper<T>::cublasGemv(cublasHandle_t handle, cublasOperation_t trans,
                           int m, int n,
                           const T           *alpha,
                           const T           *A, int lda,
                           const T           *x, int incx,
                           const T           *beta,
                           T           *y, int incy){
	return cublasSgemv( handle,  trans, m,  n, alpha, A, lda, x, incx, beta, y, incy);
};
*/
template<>
cublasStatus_t BlasWrapper<float>::cublasGemv(cublasHandle_t handle, cublasOperation_t trans,
                           int m, int n,
                           const float           *alpha,
                           const float           *A, int lda,
                           const float           *x, int incx,
                           const float           *beta,
                           float           *y, int incy){
	return cublasSgemv( handle,  trans, m,  n, alpha, A, lda, x, incx, beta, y, incy);
};

template<>
cublasStatus_t BlasWrapper<double>::cublasGemv(cublasHandle_t handle, cublasOperation_t trans,
                           int m, int n,
                           const double           *alpha,
                           const double           *A, int lda,
                           const double           *x, int incx,
                           const double           *beta,
                           double           *y, int incy){
	return cublasDgemv( handle,  trans, m,  n, alpha, A, lda, x, incx, beta, y, incy);
};


/*template<typename T>
cublasStatus_t BlasWrapper<T>::cublasGeam(cublasHandle_t handle,
                          cublasOperation_t transa, cublasOperation_t transb,
                          int m, int n,
                          const T           *alpha,
                          const T           *A, int lda,
                          const T           *beta,
                          const T           *B, int ldb,
                          T           *C, int ldc){
	return cublasSgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B,  ldb, C, ldc);

};
*/
template<>
cublasStatus_t BlasWrapper<float>::cublasGeam(cublasHandle_t handle,
                          cublasOperation_t transa, cublasOperation_t transb,
                          int m, int n,
                          const float           *alpha,
                          const float           *A, int lda,
                          const float           *beta,
                          const float           *B, int ldb,
                          float           *C, int ldc){
	return cublasSgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B,  ldb, C, ldc);
};

template<>
cublasStatus_t BlasWrapper<double>::cublasGeam(cublasHandle_t handle,
                          cublasOperation_t transa, cublasOperation_t transb,
                          int m, int n,
                          const double           *alpha,
                          const double           *A, int lda,
                          const double           *beta,
                          const double           *B, int ldb,
                          double           *C, int ldc){

	return cublasDgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B,  ldb, C, ldc);
};

template class BlasWrapper<float>;
template class BlasWrapper<double>;
