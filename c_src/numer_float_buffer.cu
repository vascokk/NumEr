#ifdef __CUDACC__
    #ifdef __CUDA_ARCH__
        #if __CUDA_ARCH__ > 120
            #define DOUBLE_PRECISION true
        #else
            #define DOUBLE_PRECISION false
        #endif
    #else
        #define DOUBLE_PRECISION false
    #endif
#else
    #define DOUBLE_PRECISION false
#endif


#include <stdio.h>
#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include "numer_buffer.h"

struct CastToFloat
{
    float operator()(double value) const { return static_cast<float>(value);}
};

template<typename T> NumerFloatBuffer<T>::NumerFloatBuffer() {
    this->h_data = new std::vector<T>();
    this->d_data = new thrust::device_vector<T>();
    this->_size = this->h_data->size();
}

template<typename T> NumerFloatBuffer<T>::NumerFloatBuffer(unsigned long size) {
    this->h_data = new std::vector<T>(size, 0);
    this->d_data = new thrust::device_vector<T>(size, 0);
    this->_size = this->h_data->size();
}

template<typename T> NumerFloatBuffer<T>::~NumerFloatBuffer() {
    delete this->h_data;
    delete this->d_data;
}

template<typename T> 
unsigned int NumerFloatBuffer<T>::size() {
    return this->_size;
}

template<typename T>
void NumerFloatBuffer<T>::write(ErlNifEnv *env, ERL_NIF_TERM data) {
    ERL_NIF_TERM head;
    double value;
    long lvalue;
    
    this->h_data->clear();    
    this->d_data->clear();
    while (enif_get_list_cell(env, data, &head, &data)) {
        if (enif_get_double(env, head, &value)) {
            this->h_data->push_back((T)value);
        }else if (enif_get_long(env, head, &lvalue)) {
            this->h_data->push_back((T)lvalue);
        }
    }

    this->_size = this->h_data->size();

    if(!DOUBLE_PRECISION){
        this->d_data->clear();
        std::transform(this->h_data->begin(), this->h_data->end(),  std::back_inserter(*(this->d_data)), CastToFloat());
    }
    else
        *(this->d_data) = *(this->h_data);

    cudaDeviceSynchronize();
}

template<typename T> 
ERL_NIF_TERM NumerFloatBuffer<T>::toErlTerms(ErlNifEnv *env) {
    typename std::vector<T>::iterator iter;

    h_data->clear();
    h_data->resize(d_data->size());
    thrust::copy(d_data->begin(), d_data->end(), this->h_data->begin());

    ERL_NIF_TERM retval = enif_make_list(env, 0.0f);
    if (h_data->size() > 0) {
        for (iter = h_data->end(); iter != h_data->begin();) {
            --iter;
            retval = enif_make_list_cell(env, enif_make_double(env, *iter), retval);
        }
    }    
    return retval;
}

template<typename T> 
void NumerFloatBuffer<T>::clear() {
    this->h_data->clear();
    this->d_data->clear();
    this->_size = this->h_data->size();
}

template class NumerFloatBuffer<float>;
template class NumerFloatBuffer<double>;
