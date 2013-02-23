#include <vector>
#include <stdio.h>
#include <iostream>
#include <algorithm>

#include "cuda.h"
#include "cublas_v2.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/binary_search.h>
#include <thrust/set_operations.h>
#include <thrust/extrema.h>

#include "numer_blas.h"
#include "numer_cublas_wrappers.h"

extern template class BlasWrapper<float>;
extern template class BlasWrapper<double>;

// Multiply the arrays A and B on GPU and save the result in C
// C(m,n) = A(m,k) * B(k,n)
template<typename T> 
void NumErBlas<T>::numer_gemm(const int transpose_a, const int transpose_b, const int m, const int n, const int k, const T alpha, thrust::device_vector<T> *d_a, thrust::device_vector<T> *d_b, const T beta, thrust::device_vector<T> *d_c){
    BlasWrapper<T> bw;

    int lda=m,ldb=k,ldc=m;

    //std::cout<<"numer_blas_cu: -1-"<<std::endl;

    cublasOperation_t _transpose_a, _transpose_b;

    switch (transpose_a){
        case 0: 
            _transpose_a = CUBLAS_OP_N;
            break;
        case 1: 
            _transpose_a = CUBLAS_OP_T;
            lda = k;
            break;
        case 2: 
            _transpose_a = CUBLAS_OP_C;
            lda = k;
            break;
    }

    switch (transpose_b){
        case 0: 
            _transpose_b = CUBLAS_OP_N;
            break;
        case 1: 
            _transpose_b = CUBLAS_OP_T;
            ldb = n;
            break;
        case 2: 
            _transpose_b = CUBLAS_OP_C;
            ldb = n;
            break;
    }

    // Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
    //std::cout<<"numer_blas_cu: -2-"<<std::endl;
    // Do the actual multiplication
    cublasStatus_t res = bw.cublasGemm(handle, _transpose_a, _transpose_b, m, n, k, &alpha, thrust::raw_pointer_cast(d_a->data()), lda, thrust::raw_pointer_cast(d_b->data()), ldb, &beta, thrust::raw_pointer_cast(d_c->data()), ldc);
    //std::cout << "\ncublasSgemm Status = " << res << std::endl;
    //std::cout<<"numer_blas_cu: -3-  res="<<res<<std::endl;
    // Destroy the handle
    cublasDestroy(handle);
}

template<typename T> 
void NumErBlas<T>::numer_gemv(const int transpose, const int m, const int n, const T alpha, thrust::device_vector<T> *d_a, thrust::device_vector<T> *d_x, const T beta, thrust::device_vector<T> *d_y){
    BlasWrapper<T> bw;

    int lda=m;
    int incx=1, incy=1;

    cublasOperation_t _transpose;

    switch (transpose){
        case 0: _transpose = CUBLAS_OP_N;break;
        case 1: _transpose = CUBLAS_OP_T;break;
        case 2: _transpose = CUBLAS_OP_C;break;
    }

    // Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Do the actual multiplication
    cublasStatus_t res = bw.cublasGemv(handle, _transpose, m, n, &alpha, thrust::raw_pointer_cast(&(*d_a)[0]), lda, thrust::raw_pointer_cast(&(*d_x)[0]), incx, &beta, thrust::raw_pointer_cast(&(*d_y)[0]), incy);

    // Destroy the handle
    cublasDestroy(handle);
}


//SAXPY:  y <- a * x + y
template<typename T>
void NumErBlas<T>::numer_saxpy(T a, thrust::device_vector<T> *d_x, thrust::device_vector<T> *d_y)
{
    //const T _a = (T)a;
    thrust::transform(d_x->begin(), d_x->end(), d_y->begin(), d_y->begin(), saxpy_functor(a));
}

//Transpose:  B<-A'
struct transpose_index : public thrust::unary_function<size_t,size_t>
{
  size_t m, n;

  __host__ __device__
  transpose_index(size_t _m, size_t _n) : m(_m), n(_n) {}

  __host__ __device__
  size_t operator()(size_t linear_index)
  {
      size_t i = linear_index / n;
      size_t j = linear_index % n;

      return m * j + i;
  }
};

template<typename T>
void NumErBlas<T>::numer_transpose(const int _m, const int _n, thrust::device_vector<T> *d_a, thrust::device_vector<T> *d_b){

    size_t m = _m; 
    size_t n = _n;
    thrust::counting_iterator<size_t> indices(0);

    thrust::gather
        (thrust::make_transform_iterator(indices, transpose_index(n, m)),
        thrust::make_transform_iterator(indices, transpose_index(n, m)) + d_b->size(),
        d_a->begin(),
        d_b->begin());    
}

template<typename T>
void NumErBlas<T>::numer_geam(const int transpose_a, const int transpose_b, const int m, const int n, const T alpha, thrust::device_vector<T> *d_a, const T beta, thrust::device_vector<T> *d_b, thrust::device_vector<T> *d_c){
    BlasWrapper<T> bw;

    int lda=m,ldb=m,ldc=m;
    cublasOperation_t _transpose_a, _transpose_b;

    switch (transpose_a){
        case 0: _transpose_a = CUBLAS_OP_N;break;
        case 1: _transpose_a = CUBLAS_OP_T;break;
        case 2: _transpose_a = CUBLAS_OP_C;break;
    }

    switch (transpose_b){
        case 0: _transpose_b = CUBLAS_OP_N;break;
        case 1: _transpose_b = CUBLAS_OP_T;break;
        case 2: _transpose_b = CUBLAS_OP_C;break;
    }
 
    // Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Do the actual multiplication
    cublasStatus_t res = bw.cublasGeam(handle, _transpose_a, _transpose_b, m, n, &alpha, thrust::raw_pointer_cast(d_a->data()), lda, &beta, thrust::raw_pointer_cast(d_b->data()), ldb, thrust::raw_pointer_cast(d_c->data()), ldc);

    // Destroy the handle
    cublasDestroy(handle);
}


template<typename T>
void NumErBlas<T>::numer_smm(const T alpha, thrust::device_vector<T> *d_a, thrust::device_vector<T> *d_b){
    thrust::transform(d_a->begin(), d_a->end(), d_b->begin(), smm_functor(alpha));
}

template<typename T>
void NumErBlas<T>::numer_mulsimp(thrust::device_vector<T> *d_a, thrust::device_vector<T> *d_b, thrust::device_vector<T> *d_c){
    thrust::transform(d_a->begin(), d_a->end(), d_b->begin(), d_c->begin(), mulsimp_functor());
}

template class NumErBlas<float>;
template class NumErBlas<double>;