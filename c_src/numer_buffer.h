#ifndef numer_BUFFER
#define numer_BUFFER

#include <string>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "erl_nif.h"

enum MatrixOrientation {
    ROW_MAJOR,
    COLUMN_MAJOR
};

enum NumerBufferTypes {
    //BUF_TYPE_INTEGER,
    BUF_TYPE_FLOAT,
   // BUF_TYPE_MATRIX_INTEGER,
    BUF_TYPE_MATRIX_FLOAT
};


class NumerBuffer {
public:
    virtual ~NumerBuffer() { };
    virtual unsigned int size() = 0;
    virtual NumerBufferTypes type() = 0;
    virtual void write(ErlNifEnv *env, ERL_NIF_TERM data) = 0;
    virtual void clear() = 0;
    virtual ERL_NIF_TERM toErlTerms(ErlNifEnv *env) = 0;   
protected:
    unsigned int _size;
};



template<typename T>
class NumerFloatBuffer : public NumerBuffer{
public:
    NumerFloatBuffer();
    NumerFloatBuffer(unsigned long size);
    virtual ~NumerFloatBuffer();
    virtual unsigned int size();
    virtual NumerBufferTypes type() { return BUF_TYPE_FLOAT; };
    virtual ERL_NIF_TERM toErlTerms(ErlNifEnv *env);
    virtual void write(ErlNifEnv *env, ERL_NIF_TERM data);
    virtual void clear();
    std::vector<T>* get_host_data(){return h_data;};
    thrust::device_vector<T>* get_data(){return d_data;};
protected:
    std::vector<T> *h_data;
    thrust::device_vector<T> *d_data;
};

class NumerMatrixBuffer{
public:
/*    virtual unsigned int size() = 0;
    virtual NumerBufferTypes type() = 0;
    virtual void write(ErlNifEnv *env, ERL_NIF_TERM data) = 0;
    virtual void clear() = 0;
    virtual ERL_NIF_TERM toErlTerms(ErlNifEnv *env) = 0;       
*/    unsigned int rows() {return _rows;};
    unsigned int cols() {return _cols;};
    MatrixOrientation storage() {return orientation;}
protected:
    unsigned int _rows;
    unsigned int _cols;
    MatrixOrientation orientation;
};

template<typename T>
class NumerMatrixFloatBuffer : public NumerMatrixBuffer, public NumerFloatBuffer<T> {
public:
    NumerMatrixFloatBuffer();
    NumerMatrixFloatBuffer(unsigned int rows, unsigned int cols, MatrixOrientation orientation);
    virtual ~NumerMatrixFloatBuffer();
    virtual unsigned int size();
    virtual NumerBufferTypes type() { return BUF_TYPE_MATRIX_FLOAT; };
    virtual ERL_NIF_TERM toErlTerms(ErlNifEnv *env);
    virtual void write(ErlNifEnv *env, ERL_NIF_TERM data);
    virtual void clear();
};

#endif
