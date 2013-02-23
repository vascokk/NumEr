#ifndef numer_ML
#define numer_ML

#include <vector>
#include "cublas_v2.h"

template<typename T>
class Ml{
public:
	struct saxpy_functor
	{
	    const T a;
	    saxpy_functor(T _a) : a(_a) {}
	    __host__ __device__
	        T operator()(const T& x, const T& y) const { 
	            return a * x + y;
	        }
	};
	void numer_gd(thrust::device_vector<T> *d_theta, thrust::device_vector<T> *d_x, thrust::device_vector<T> *d_y, const unsigned int num_features, const unsigned int num_samples);
	void numer_gd_learn(thrust::device_vector<T> *d_theta, thrust::device_vector<T> *d_x, thrust::device_vector<T> *d_y, const unsigned int num_features, const unsigned int num_samples, const float learning_rate, const unsigned int iterations);
	void gradient_descent(cublasHandle_t handle, thrust::device_vector<T> *d_theta, thrust::device_vector<T> *d_x, thrust::device_vector<T> *d_y, const unsigned int num_features, const unsigned int num_samples);
};

#endif