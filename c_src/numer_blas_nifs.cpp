#include <stdio.h>
#include <iostream>
#include "cublas_v2.h"


#include "numer.h"
#include "numer_buffer.h"
#include "numer_blas.h"

extern template class NumErBlas<float>;
extern template class NumErBlas<double>;

//TODO class instance

///////////////////Matrix operations
// C(m,n) = A(m,k) * B(k,n)
//gemm(_Ctx, _transpose_op_A, _transpose_op_B, _m, _n, _k, _alpha, _A, _B, _beta, _C ) 
ERL_NIF_TERM numer_nifs_gemm(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    NumerContextRef *ctxRef;
    NumerBufferRef *ref_A, *ref_B, *ref_C;
    //NumerBufferRef *ref_C;
    unsigned long transpose_a, transpose_b;
    unsigned long  m, n, k;
    double alpha, beta;

    if (argc != 11 || !enif_get_resource(env, argv[0], numer_context_resource, (void **) &ctxRef) ||
        !enif_get_ulong(env, argv[1], &transpose_a)||
        !enif_get_ulong(env, argv[2], &transpose_b)||
        !enif_get_ulong(env, argv[3], &m)||
        !enif_get_ulong(env, argv[4], &n)||
        !enif_get_ulong(env, argv[5], &k)||
        !enif_get_double(env, argv[6], &alpha)||
        !enif_get_resource(env, argv[7], numer_buffer_resource, (void **) &ref_A) ||
        !enif_get_resource(env, argv[8], numer_buffer_resource, (void **) &ref_B)||
        !enif_get_double(env, argv[9], &beta)||
        !enif_get_resource(env, argv[10], numer_buffer_resource, (void **) &ref_C)
        ) {
        return enif_make_badarg(env);
    }


    if(transpose_a == CUBLAS_OP_N){
        if(((NumerMatrixFloatBuffer<float>*)ref_A->buffer)->rows() != m || ((NumerMatrixFloatBuffer<float>*)ref_A->buffer)->cols() != k){
            return enif_make_tuple2(env, ATOM_ERROR, enif_make_atom(env, "Matrix A dimensions do not match m,k parameters")); 
        }
    }else{
         if(((NumerMatrixFloatBuffer<float>*)ref_A->buffer)->rows() != k || ((NumerMatrixFloatBuffer<float>*)ref_A->buffer)->cols() != n){
            return enif_make_tuple2(env, ATOM_ERROR, enif_make_atom(env, "Matrix A dimensions do not match m,k parameters")); 
        }
    }

    if(transpose_b == CUBLAS_OP_N){
        if(((NumerMatrixFloatBuffer<float>*)ref_B->buffer)->rows() != k || ((NumerMatrixFloatBuffer<float>*)ref_B->buffer)->cols() != n){
            return enif_make_tuple2(env, ATOM_ERROR, enif_make_atom(env, "Matrix B dimensions do not match k,n parameters")); 
        }
    }else{
        if(((NumerMatrixFloatBuffer<float>*)ref_B->buffer)->rows() != n || ((NumerMatrixFloatBuffer<float>*)ref_B->buffer)->cols() != k){
            return enif_make_tuple2(env, ATOM_ERROR, enif_make_atom(env, "Matrix B dimensions do not match k,n parameters")); 
        }
    }    


    if(((NumerMatrixFloatBuffer<float>*)ref_C->buffer)->rows() != m || ((NumerMatrixFloatBuffer<float>*)ref_C->buffer)->cols() != n){
        return enif_make_tuple2(env, ATOM_ERROR, enif_make_atom(env, "Matrix C dimensions do not match m,n parameters")); 
    }        

  
    cuCtxSetCurrent(ctxRef->ctx);

    if(ctxRef->doublePrecision){
        NumErBlas<double> blas;
        blas.numer_gemm(transpose_a, transpose_b, m, n, k, alpha, ((NumerMatrixFloatBuffer<double>*)ref_A->buffer)->get_data(), ((NumerMatrixFloatBuffer<double>*)ref_B->buffer)->get_data(), beta, ((NumerMatrixFloatBuffer<double>*)ref_C->buffer)->get_data());    
    }else{
        NumErBlas<float> blas;
        blas.numer_gemm(transpose_a, transpose_b, m, n, k, alpha, ((NumerMatrixFloatBuffer<float>*)ref_A->buffer)->get_data(), ((NumerMatrixFloatBuffer<float>*)ref_B->buffer)->get_data(), beta, ((NumerMatrixFloatBuffer<float>*)ref_C->buffer)->get_data());
    }
    
    
   /* NumErBlas<float> blas;
    blas.numer_gemm(transpose_a, transpose_b, m, n, k, alpha, ((NumerMatrixFloatBuffer<float>*)ref_A->buffer)->get_data(), ((NumerMatrixFloatBuffer<float>*)ref_B->buffer)->get_data(), beta, ((NumerMatrixFloatBuffer<float>*)ref_C->buffer)->get_data());
*/
    return ATOM_OK;
}

//numer_nifs:gemv(Ctx, _m, _n, _alpha, A, X, _betha, Y),
ERL_NIF_TERM numer_nifs_gemv(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    NumerContextRef *ctxRef;
    NumerMatrixBufferRef *ref_A;
    NumerBufferRef *ref_X, *ref_Y;
    unsigned long transpose;
    unsigned long  m, n;
    double alpha, beta;

    if (argc != 9 || 
        !enif_get_resource(env, argv[0], numer_context_resource, (void **) &ctxRef) ||
        !enif_get_ulong(env, argv[1], &transpose)||
        !enif_get_ulong(env, argv[2], &m)||
        !enif_get_ulong(env, argv[3], &n)||
        !enif_get_double(env, argv[4], &alpha)||
        !enif_get_resource(env, argv[5], numer_buffer_resource, (void **) &ref_A) ||
        !enif_get_resource(env, argv[6], numer_buffer_resource, (void **) &ref_X)||
        !enif_get_double(env, argv[7], &beta)||
        !enif_get_resource(env, argv[8], numer_buffer_resource, (void **) &ref_Y)) {

        return enif_make_badarg(env);
    }

    if(ref_A->buffer->rows() != m || ref_A->buffer->cols() != n){
        return enif_make_tuple2(env, ATOM_ERROR, enif_make_atom(env, "Matrix A dimensions do not match m,n parameters")); 
    }

    cuCtxSetCurrent(ctxRef->ctx);
    if(ctxRef->doublePrecision){
        NumErBlas<double> blas;
        blas.numer_gemv(transpose, m, n, alpha, ((NumerMatrixFloatBuffer<double> *)ref_A->buffer)->get_data(), ((NumerFloatBuffer<double> *)ref_X->buffer)->get_data(), beta, ((NumerFloatBuffer<double> *)ref_Y->buffer)->get_data());
    }else{
        NumErBlas<float> blas;
        blas.numer_gemv(transpose, m, n, alpha, ((NumerMatrixFloatBuffer<float> *)ref_A->buffer)->get_data(), ((NumerFloatBuffer<float> *)ref_X->buffer)->get_data(), beta, ((NumerFloatBuffer<float> *)ref_Y->buffer)->get_data());
    }
    //blas->numer_gemv(transpose, m, n, alpha, ((NumerMatrixFloatBuffer *)ref_A->buffer)->get_data(), ((NumerFloatBuffer *)ref_X->buffer)->get_data(), beta, ((NumerFloatBuffer *)ref_Y->buffer)->get_data());
    //delete blas;

    return ATOM_OK;
}

ERL_NIF_TERM numer_nifs_saxpy(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    NumerContextRef *ctxRef;
    NumerBufferRef *ref_X, *ref_Y;
    double a;

    if (argc != 4 || 
        !enif_get_resource(env, argv[0], numer_context_resource, (void **) &ctxRef) ||
        !enif_get_double(env, argv[1], &a)||
        !enif_get_resource(env, argv[2], numer_buffer_resource, (void **) &ref_X)||
        !enif_get_resource(env, argv[3], numer_buffer_resource, (void **) &ref_Y)) {

        return enif_make_badarg(env);
    }

    if(ref_X->buffer->size() != ref_Y->buffer->size()){
        return enif_make_tuple2(env, ATOM_ERROR, enif_make_atom(env, "Size X does not match size Y.")); 
    }

    cuCtxSetCurrent(ctxRef->ctx);
    if(ctxRef->doublePrecision){
        NumErBlas<double> blas;
        blas.numer_saxpy(a, ((NumerMatrixFloatBuffer<double> *)ref_X->buffer)->get_data(), ((NumerMatrixFloatBuffer<double> *)ref_Y->buffer)->get_data());
        //blas.numer_saxpy(a, ((NumerFloatBuffer<double> *)ref_X->buffer)->get_data(), ((NumerFloatBuffer<double> *)ref_Y->buffer)->get_data());
    }else{
        NumErBlas<float> blas;
        blas.numer_saxpy(a, ((NumerMatrixFloatBuffer<float> *)ref_X->buffer)->get_data(), ((NumerMatrixFloatBuffer<float> *)ref_Y->buffer)->get_data());
        //blas.numer_saxpy(a, ((NumerFloatBuffer<float> *)ref_X->buffer)->get_data(), ((NumerFloatBuffer<float> *)ref_Y->buffer)->get_data());
    }
    //blas->numer_saxpy(a, ((NumerFloatBuffer *)ref_X->buffer)->get_data(), ((NumerFloatBuffer *)ref_Y->buffer)->get_data());
    //delete blas;

    return ATOM_OK;
}

ERL_NIF_TERM numer_nifs_transpose(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    NumerContextRef *ctxRef;
    NumerMatrixBufferRef *ref_A, *ref_B;
    unsigned long m, n;

    if (argc != 3 || 
        !enif_get_resource(env, argv[0], numer_context_resource, (void **) &ctxRef) ||
        !enif_get_resource(env, argv[1], numer_buffer_resource, (void **) &ref_A)||
        !enif_get_resource(env, argv[2], numer_buffer_resource, (void **) &ref_B)) {

        return enif_make_badarg(env);
    }

    if(ref_A->buffer->rows() != ref_B->buffer->cols() ||
        ref_A->buffer->cols() != ref_B->buffer->rows() ){
        return enif_make_tuple2(env, ATOM_ERROR, enif_make_atom(env, "Size A does not match the transpose size B.")); 
    }

    m = ref_A->buffer->rows();
    n = ref_A->buffer->cols();
    
    cuCtxSetCurrent(ctxRef->ctx);
    if(ctxRef->doublePrecision){
        NumErBlas<double> blas;
        blas.numer_transpose(n, m, ((NumerMatrixFloatBuffer<double> *)ref_A->buffer)->get_data(), ((NumerMatrixFloatBuffer<double> *)ref_B->buffer)->get_data());
    }else{
        NumErBlas<float> blas;
        blas.numer_transpose(n, m, ((NumerMatrixFloatBuffer<float> *)ref_A->buffer)->get_data(), ((NumerMatrixFloatBuffer<float> *)ref_B->buffer)->get_data());
    }
    //blas->numer_transpose(n, m, ((NumerMatrixFloatBuffer *)ref_A->buffer)->get_data(), ((NumerMatrixFloatBuffer *)ref_B->buffer)->get_data());
    //delete blas;

    return ATOM_OK;
}


ERL_NIF_TERM numer_nifs_geam(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    NumerContextRef *ctxRef;
    NumerBufferRef *ref_A, *ref_B, *ref_C;
    unsigned long transpose_a, transpose_b;
    unsigned long  m, n;
    double alpha, beta;

    if (argc != 10 || !enif_get_resource(env, argv[0], numer_context_resource, (void **) &ctxRef) ||
        !enif_get_ulong(env, argv[1], &transpose_a)||
        !enif_get_ulong(env, argv[2], &transpose_b)||
        !enif_get_ulong(env, argv[3], &m)||
        !enif_get_ulong(env, argv[4], &n)||
        !enif_get_double(env, argv[5], &alpha)||
        !enif_get_resource(env, argv[6], numer_buffer_resource, (void **) &ref_A) ||
        !enif_get_double(env, argv[7], &beta)||
        !enif_get_resource(env, argv[8], numer_buffer_resource, (void **) &ref_B)||
        !enif_get_resource(env, argv[9], numer_buffer_resource, (void **) &ref_C)
        ) {
        return enif_make_badarg(env);
    }

    if(transpose_a == CUBLAS_OP_N){
        if(((NumerMatrixFloatBuffer<float>*)ref_A->buffer)->rows() != m ){
            return enif_make_tuple2(env, ATOM_ERROR, enif_make_atom(env, "Matrix A rows do not match the m parameter")); 
        }
        if(((NumerMatrixFloatBuffer<float>*)ref_A->buffer)->cols() != n ){
            return enif_make_tuple2(env, ATOM_ERROR, enif_make_atom(env, "Matrix A columns do not match the n parameter")); 
        }
    }else{
        if(((NumerMatrixFloatBuffer<float>*)ref_A->buffer)->rows() != n ){
            return enif_make_tuple2(env, ATOM_ERROR, enif_make_atom(env, "Matrix A rows do not match the n parameter")); 
        }
        if(((NumerMatrixFloatBuffer<float>*)ref_A->buffer)->cols() != m ){
            return enif_make_tuple2(env, ATOM_ERROR, enif_make_atom(env, "Matrix A columns do not match the m parameter")); 
        }
    }

    if(transpose_b == CUBLAS_OP_N){        
        if(((NumerMatrixFloatBuffer<float>*)ref_B->buffer)->rows()!= m){
            return enif_make_tuple2(env, ATOM_ERROR, enif_make_atom(env, "Matrix B rows do not match m parameters")); 
        }    
        if(((NumerMatrixFloatBuffer<float>*)ref_B->buffer)->cols() != n){
            return enif_make_tuple2(env, ATOM_ERROR, enif_make_atom(env, "Matrix B columns do not match n parameter")); 
        }
    }else{
        if(((NumerMatrixFloatBuffer<float>*)ref_B->buffer)->rows()!= n){
            return enif_make_tuple2(env, ATOM_ERROR, enif_make_atom(env, "Matrix B rows do not match n parameters")); 
        }
        if(((NumerMatrixFloatBuffer<float>*)ref_B->buffer)->cols()!= m){
            return enif_make_tuple2(env, ATOM_ERROR, enif_make_atom(env, "Matrix B columns do not match m parameters")); 
        }    
    }

    if(((NumerMatrixFloatBuffer<float>*)ref_C->buffer)->rows() != m || ((NumerMatrixFloatBuffer<float>*)ref_C->buffer)->cols() != n){
        return enif_make_tuple2(env, ATOM_ERROR, enif_make_atom(env, "Matrix C dimensions do not match m,n parameters")); 
    }


    cuCtxSetCurrent(ctxRef->ctx);
    if(ctxRef->doublePrecision){
        NumErBlas<double> blas;
        blas.numer_geam(transpose_a, transpose_b, m, n, alpha, ((NumerMatrixFloatBuffer<double>*)ref_A->buffer)->get_data(), beta, ((NumerMatrixFloatBuffer<double>*)ref_B->buffer)->get_data(), ((NumerMatrixFloatBuffer<double>*)ref_C->buffer)->get_data());
    }else{
        NumErBlas<float> blas;
        blas.numer_geam(transpose_a, transpose_b, m, n, alpha, ((NumerMatrixFloatBuffer<float>*)ref_A->buffer)->get_data(), beta, ((NumerMatrixFloatBuffer<float>*)ref_B->buffer)->get_data(), ((NumerMatrixFloatBuffer<float>*)ref_C->buffer)->get_data());
    }

    //blas->numer_geam(transpose_a, transpose_b, m, n, alpha, ((NumerMatrixFloatBuffer*)ref_A->buffer)->get_data(), beta, ((NumerMatrixFloatBuffer*)ref_B->buffer)->get_data(), ((NumerMatrixFloatBuffer*)ref_C->buffer)->get_data());
    //delete blas;

    return ATOM_OK;
}

ERL_NIF_TERM numer_nifs_smm(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    NumerContextRef *ctxRef;
    NumerBufferRef *ref_A, *ref_B;
    double alpha;
    

    if (argc != 4 || !enif_get_resource(env, argv[0], numer_context_resource, (void **) &ctxRef) ||
        !enif_get_double(env, argv[1], &alpha)||
        !enif_get_resource(env, argv[2], numer_buffer_resource, (void **) &ref_A) ||
        !enif_get_resource(env, argv[3], numer_buffer_resource, (void **) &ref_B)
        ) {
        return enif_make_badarg(env);
    }

    if(ref_A->buffer->size() != ref_B->buffer->size() ){
        return enif_make_tuple2(env, ATOM_ERROR, enif_make_atom(env, "Buffer A size does not match buffer B size")); 
    }

    cuCtxSetCurrent(ctxRef->ctx);
    
    if(ctxRef->doublePrecision){
        NumErBlas<double> blas;
        blas.numer_smm(alpha, ((NumerMatrixFloatBuffer<double>*)ref_A->buffer)->get_data(), ((NumerMatrixFloatBuffer<double>*)ref_B->buffer)->get_data());
    }else{
        NumErBlas<float> blas;
        blas.numer_smm(alpha, ((NumerMatrixFloatBuffer<float>*)ref_A->buffer)->get_data(), ((NumerMatrixFloatBuffer<float>*)ref_B->buffer)->get_data());
    }

    return ATOM_OK;
}


ERL_NIF_TERM numer_nifs_mulsimp(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    NumerContextRef *ctxRef;
    NumerBufferRef *ref_A, *ref_B, *ref_C;

    if (argc != 4 || !enif_get_resource(env, argv[0], numer_context_resource, (void **) &ctxRef) ||
        !enif_get_resource(env, argv[1], numer_buffer_resource, (void **) &ref_A) ||
        !enif_get_resource(env, argv[2], numer_buffer_resource, (void **) &ref_B) ||
        !enif_get_resource(env, argv[3], numer_buffer_resource, (void **) &ref_C)
        ) {
        return enif_make_badarg(env);
    }

    if(ref_A->buffer->size() != ref_B->buffer->size() ){
        return enif_make_tuple2(env, ATOM_ERROR, enif_make_atom(env, "Buffer A size does not match buffer B size")); 
    }

    cuCtxSetCurrent(ctxRef->ctx);
    
    if(ctxRef->doublePrecision){
        NumErBlas<double> blas;
        blas.numer_mulsimp(((NumerMatrixFloatBuffer<double>*)ref_A->buffer)->get_data(), ((NumerMatrixFloatBuffer<double>*)ref_B->buffer)->get_data(),  ((NumerMatrixFloatBuffer<double>*)ref_C->buffer)->get_data());
    }else{
        NumErBlas<float> blas;
        blas.numer_mulsimp(((NumerMatrixFloatBuffer<float>*)ref_A->buffer)->get_data(), ((NumerMatrixFloatBuffer<float>*)ref_B->buffer)->get_data(), ((NumerMatrixFloatBuffer<float>*)ref_C->buffer)->get_data());
    }

    return ATOM_OK;
}
