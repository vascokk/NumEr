#include "cuda.h"
#include "cublas_v2.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <stdio.h>
#include <iostream>

#include "numer_kernels.h"
#include "numer_blas.h"
#include "numer_cublas_wrappers.h"
#include "numer_ml.h"

extern template class BlasWrapper<float>;
extern template class BlasWrapper<double>;
extern template class NumErBlas<float>;
extern template class NumErBlas<double>;
extern template class Kernels<float>;
extern template class Kernels<double>;

template<typename T>
void Ml<T>::gradient_descent(cublasHandle_t handle, thrust::device_vector<T> *d_theta, thrust::device_vector<T> *d_x, thrust::device_vector<T> *d_y, const unsigned int num_features, const unsigned int num_samples)
{
    BlasWrapper<T> bw;
    /*Kernels<T> kernels;
    NumErBlas<T> blas;*/
    int lda,ldb,ldc;

    // Create a handle for CUBLAS
    //cublasStatus_t res;

    //Grad = (1/m)* ( X * (sigmoid(Theta*X) - Y) )
    // tmp1 = gemm(1*Theta*X + 0*H)
    lda=1,ldb=num_samples, ldc=1;
    thrust::device_vector<T> d_tmp1(num_samples, 0.0);
    const T alf = 1.0;
    const T bet = 0.0;
    //const float *_alpha = &alf;
    //const float *_beta =  &bet;

    bw.cublasGemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 1, num_samples, num_features, &alf, thrust::raw_pointer_cast(d_theta->data()), lda, thrust::raw_pointer_cast(d_x->data()), ldb, &bet, thrust::raw_pointer_cast(d_tmp1.data()), ldc);
    //% H=sigmoid(Theta*X)
    thrust::device_vector<T> d_h(d_y->size());
    thrust::transform(d_tmp1.begin(), d_tmp1.end(), d_h.begin(), typename Kernels<T>::sigmoid());
    //%H - Y
    thrust::transform(d_y->begin(), d_y->end(), d_h.begin(), d_h.begin(), typename NumErBlas<T>::saxpy_functor(-1.0));
    const T alpha2 = 1.0/num_samples;    
    //const float *alpha2 = &alf2;
    bw.cublasGemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, num_features, num_samples, &alpha2, thrust::raw_pointer_cast(d_h.data()), lda, thrust::raw_pointer_cast(d_x->data()), ldb, &bet, thrust::raw_pointer_cast(d_theta->data()), ldc);
}

template<typename T>
void Ml<T>::numer_gd(thrust::device_vector<T> *d_theta, thrust::device_vector<T> *d_x, thrust::device_vector<T> *d_y, const unsigned int num_features, const unsigned int num_samples)
{
    // Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    gradient_descent(handle, d_theta, d_x, d_y, num_features, num_samples);

    // Destroy the handle
    cublasDestroy(handle);
}

template<typename T>
void Ml<T>::numer_gd_learn(thrust::device_vector<T> *d_theta, thrust::device_vector<T> *d_x, thrust::device_vector<T> *d_y, const unsigned int num_features, const unsigned int num_samples, const float learning_rate, const unsigned int iterations){
    //NumErBlas<T> blas;
    cublasHandle_t handle;
    cublasCreate(&handle);

    thrust::device_vector<T> d_theta_tmp = *d_theta;

    for(int i=0; i<iterations; i++){
        gradient_descent(handle, d_theta, d_x, d_y, num_features, num_samples);
        thrust::transform(d_theta->begin(), d_theta->end(), d_theta_tmp.begin(), d_theta_tmp.begin(),  typename NumErBlas<T>::saxpy_functor(-learning_rate));
        *d_theta = d_theta_tmp;
    }
    // Destroy the handle
    cublasDestroy(handle);
}

template class Ml<float>;
template class Ml<double>;