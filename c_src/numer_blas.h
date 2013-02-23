#ifndef numer_BLAS
#define numer_BLAS

#include <vector>

template<typename T>
class NumErBlas
{
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

	struct smm_functor
	{
	    const T a;
	    smm_functor(T _a) : a(_a) {}
	    __host__ __device__
	        T operator()(const T& x) const { 
	            return a * x;
	        }
	};

	struct mulsimp_functor
	{
	    // const T a;
	    // smm_functor(T _a) : a(_a) {}
	    __host__ __device__
	        T operator()(const T& a, const T& b) const { 
	            return a * b;
	        }
	};

	void numer_gemm(const int transpose_a, const int transpose_b, const int m, const int n, const int k, const T alpha, thrust::device_vector<T> *d_a, thrust::device_vector<T> *d_b, const T beta, thrust::device_vector<T> *d_c);
	void numer_gemv(const int transpose, const int m, const int n, const T alpha, thrust::device_vector<T> *d_a, thrust::device_vector<T> *d_x,const T beta, thrust::device_vector<T> *d_y);
	void numer_saxpy(T a, thrust::device_vector<T> *d_x, thrust::device_vector<T> *d_y);
	void numer_transpose(const int m, const int n, thrust::device_vector<T> *d_a, thrust::device_vector<T> *d_b);
	void numer_geam(const int transpose_a, const int transpose_b, const int m, const int n, const T alpha, thrust::device_vector<T> *d_a, const T beta, thrust::device_vector<T> *d_b, thrust::device_vector<T> *d_c);
	void numer_smm(const T alpha, thrust::device_vector<T> *d_a, thrust::device_vector<T> *d_b);
	void numer_mulsimp(thrust::device_vector<T> *d_a, thrust::device_vector<T> *d_b, thrust::device_vector<T> *d_c);
};
#endif