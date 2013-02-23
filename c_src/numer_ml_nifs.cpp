#include "numer.h"
#include "numer_buffer.h"

#include <stdio.h>
#include <iostream>

#include "numer_ml.h"

extern template class NumErBlas<float>;
extern template class NumErBlas<double>;
extern template class Ml<float>;
extern template class Ml<double>;
extern template class NumerFloatBuffer<float>;
extern template class NumerFloatBuffer<double>;
extern template class NumerMatrixFloatBuffer<float>;
extern template class NumerMatrixFloatBuffer<double>;

ERL_NIF_TERM numer_ml_gd(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    NumerContextRef *ctxRef;
    NumerMatrixBufferRef *ref_Theta, *ref_X, *ref_Y;
    unsigned long num_features, num_samples;
    
    if (argc != 6 || !enif_get_resource(env, argv[0], numer_context_resource, (void **) &ctxRef) ||
        !enif_get_resource(env, argv[1], numer_buffer_resource, (void **) &ref_Theta) ||
        !enif_get_resource(env, argv[2], numer_buffer_resource, (void **) &ref_X) ||
        !enif_get_resource(env, argv[3], numer_buffer_resource, (void **) &ref_Y) ||
        !enif_get_ulong(env, argv[4], &num_features) ||
        !enif_get_ulong(env, argv[5], &num_samples)

        ) {
        return enif_make_badarg(env);
    }

    cuCtxSetCurrent(ctxRef->ctx);

    
    if(ctxRef->doublePrecision){
        Ml<double> ml;
        ml.numer_gd(((NumerMatrixFloatBuffer<double>*)ref_Theta->buffer)->get_data(), ((NumerMatrixFloatBuffer<double>*)ref_X->buffer)->get_data(), ((NumerMatrixFloatBuffer<double>*)ref_Y->buffer)->get_data(), num_features, num_samples);
    }else{
        Ml<float> ml;
        ml.numer_gd(((NumerMatrixFloatBuffer<float>*)ref_Theta->buffer)->get_data(), ((NumerMatrixFloatBuffer<float>*)ref_X->buffer)->get_data(), ((NumerMatrixFloatBuffer<float>*)ref_Y->buffer)->get_data(), num_features, num_samples);
    }    
    return ATOM_OK;
}

ERL_NIF_TERM numer_ml_gd_learn(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    NumerContextRef *ctxRef;
    NumerBufferRef *ref_Theta, *ref_X, *ref_Y;
    unsigned long num_features; 
    unsigned long num_samples;
    unsigned long iterations;
    double learning_rate;
    
    if (argc != 8 || !enif_get_resource(env, argv[0], numer_context_resource, (void **) &ctxRef) ||
        !enif_get_resource(env, argv[1], numer_buffer_resource, (void **) &ref_Theta) ||
        !enif_get_resource(env, argv[2], numer_buffer_resource, (void **) &ref_X) ||
        !enif_get_resource(env, argv[3], numer_buffer_resource, (void **) &ref_Y) ||
        !enif_get_ulong(env, argv[4], &num_features) ||
        !enif_get_ulong(env, argv[5], &num_samples) ||
        !enif_get_double(env, argv[6], &learning_rate) ||
        !enif_get_ulong(env, argv[7], &iterations)

        ) {
        return enif_make_badarg(env);
    }

    cuCtxSetCurrent(ctxRef->ctx);

    
    if(ctxRef->doublePrecision){
        Ml<double> ml;
        ml.numer_gd_learn(((NumerMatrixFloatBuffer<double>*)ref_Theta->buffer)->get_data(), ((NumerMatrixFloatBuffer<double>*)ref_X->buffer)->get_data(), ((NumerMatrixFloatBuffer<double>*)ref_Y->buffer)->get_data(), num_features, num_samples, (float)learning_rate, iterations);
    }else{
        Ml<float> ml;
        ml.numer_gd_learn(((NumerMatrixFloatBuffer<float>*)ref_Theta->buffer)->get_data(), ((NumerMatrixFloatBuffer<float>*)ref_X->buffer)->get_data(), ((NumerMatrixFloatBuffer<float>*)ref_Y->buffer)->get_data(), num_features, num_samples, (float)learning_rate, iterations);
    }    
    return ATOM_OK;
}