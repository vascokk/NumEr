#include <thrust/version.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/random.h>
#include <thrust/set_operations.h>
#include <thrust/extrema.h>
#include <thrust/copy.h>
#include <thrust/sort.h>

//#include <cublas.h>

#include <iostream>
#include <cmath>
#include <vector>

#include "numer_kernels.h"

template<typename T>
void  Kernels<T>::numer_sigmoid(thrust::device_vector<T> *d_a, thrust::device_vector<T> *d_b){
  thrust::transform(d_a->begin(), d_a->end(), d_b->begin(), sigmoid());
}

template<typename T>
void  Kernels<T>::numer_tanh(thrust::device_vector<T> *d_a, thrust::device_vector<T> *d_b){
  thrust::transform(d_a->begin(), d_a->end(), d_b->begin(), sigmoid2());
}

template<typename T>
void  Kernels<T>::numer_log(thrust::device_vector<T> *d_a, thrust::device_vector<T> *d_b){
  thrust::transform(d_a->begin(), d_a->end(), d_b->begin(), log_func());
}

template class Kernels<float>;
template class Kernels<double>;