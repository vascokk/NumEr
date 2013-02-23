
#include "numer.h"
#include "numer_buffer.h"
#include "numer_kernels.h"


#include <stdio.h>
#include <iostream>

extern template class Kernels<float>;
extern template class Kernels<double>;
extern template class NumerFloatBuffer<float>;
extern template class NumerFloatBuffer<double>;
extern template class NumerMatrixFloatBuffer<float>;
extern template class NumerMatrixFloatBuffer<double>;

ERL_NIF_TERM numer_nifs_sigmoid(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    NumerContextRef *ctxRef;
    NumerBufferRef *ref_A, *ref_B;

    if (argc != 3 || !enif_get_resource(env, argv[0], numer_context_resource, (void **) &ctxRef) ||
        !enif_get_resource(env, argv[1], numer_buffer_resource, (void **) &ref_A) ||
        !enif_get_resource(env, argv[2], numer_buffer_resource, (void **) &ref_B)
        ) {
        return enif_make_badarg(env);
    }

  
    if(ref_A->buffer->size() != ref_B->buffer->size() ){
        return enif_make_tuple2(env, ATOM_ERROR, enif_make_atom(env, "Buffer A size does not match buffer B size")); 
    }

    cuCtxSetCurrent(ctxRef->ctx);


    if(ctxRef->doublePrecision){
        Kernels<double> kernels;
        //kernels.numer_sigmoid(((NumerMatrixFloatBuffer<double>*)ref_A->buffer)->get_data(), ((NumerMatrixFloatBuffer<double>*)ref_B->buffer)->get_data());
        kernels.numer_sigmoid(((NumerFloatBuffer<double>*)ref_A->buffer)->get_data(), ((NumerFloatBuffer<double>*)ref_B->buffer)->get_data());
    }else{
        Kernels<float> kernels;
        //kernels.numer_sigmoid(((NumerMatrixFloatBuffer<float>*)ref_A->buffer)->get_data(), ((NumerMatrixFloatBuffer<float>*)ref_B->buffer)->get_data());
        kernels.numer_sigmoid(((NumerFloatBuffer<float>*)ref_A->buffer)->get_data(), ((NumerFloatBuffer<float>*)ref_B->buffer)->get_data());
    }
    return ATOM_OK;
}



ERL_NIF_TERM numer_nifs_tanh(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    NumerContextRef *ctxRef;
    NumerBufferRef *ref_A, *ref_B;
    
    if (argc != 3 || !enif_get_resource(env, argv[0], numer_context_resource, (void **) &ctxRef) ||
        !enif_get_resource(env, argv[1], numer_buffer_resource, (void **) &ref_A) ||
        !enif_get_resource(env, argv[2], numer_buffer_resource, (void **) &ref_B)
        ) {
        return enif_make_badarg(env);
    }

    if(ref_A->buffer->size() != ref_B->buffer->size()){
        return enif_make_tuple2(env, ATOM_ERROR, enif_make_atom(env, "Buffer A size does not match buffer B size")); 
    }

    cuCtxSetCurrent(ctxRef->ctx);
    
    if(ctxRef->doublePrecision){
        Kernels<double> kernels;
        kernels.numer_tanh(((NumerFloatBuffer<double>*)ref_A->buffer)->get_data(), ((NumerFloatBuffer<double>*)ref_B->buffer)->get_data());
    }else{
        Kernels<float> kernels;
        kernels.numer_tanh(((NumerFloatBuffer<float>*)ref_A->buffer)->get_data(), ((NumerFloatBuffer<float>*)ref_B->buffer)->get_data());
    }
    return ATOM_OK;
}

ERL_NIF_TERM numer_nifs_log(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    NumerContextRef *ctxRef;
    NumerBufferRef *ref_A, *ref_B;
    
    if (argc != 3 || !enif_get_resource(env, argv[0], numer_context_resource, (void **) &ctxRef) ||
        !enif_get_resource(env, argv[1], numer_buffer_resource, (void **) &ref_A) ||
        !enif_get_resource(env, argv[2], numer_buffer_resource, (void **) &ref_B)
        ) {
        return enif_make_badarg(env);
    }

    if(ref_A->buffer->size() != ref_B->buffer->size()){
        return enif_make_tuple2(env, ATOM_ERROR, enif_make_atom(env, "Buffer A size does not match buffer B size")); 
    }

    cuCtxSetCurrent(ctxRef->ctx);

    

    if(ctxRef->doublePrecision){
        Kernels<double> kernels;
        kernels.numer_log(((NumerFloatBuffer<double>*)ref_A->buffer)->get_data(), ((NumerFloatBuffer<double>*)ref_B->buffer)->get_data());
    }else{
        Kernels<float> kernels;
        kernels.numer_log(((NumerFloatBuffer<float>*)ref_A->buffer)->get_data(), ((NumerFloatBuffer<float>*)ref_B->buffer)->get_data());
    }    
    return ATOM_OK;
}
