#ifndef numer_KERNELS
#define numer_KERNELS

#include <vector>
#include "cuda.h"
#include "cuda_runtime.h"

template<typename T>
class Kernels{
public:
	struct sigmoid{
		__host__ __device__ T operator()(const T &x){return 1 / (1 + exp(-x));}
	};

	struct  sigmoid2{
	  __host__ __device__ T operator()(const T &x){return tanh(x);}
	};

	struct  log_func{
	  __host__ __device__ T operator()(const T &x){return log(x);}
	};
	void numer_sigmoid(thrust::device_vector<T> *d_a, thrust::device_vector<T> *d_b);
	void numer_tanh(thrust::device_vector<T> *d_a, thrust::device_vector<T> *d_b);
	void numer_log(thrust::device_vector<T> *d_a, thrust::device_vector<T> *d_b);
};
#endif