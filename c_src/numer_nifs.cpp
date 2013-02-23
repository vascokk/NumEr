// -------------------------------------------------------------------
//
// numer: An Erlang framework for performing CUDA-enabled operations
//
// Copyright (c) 2011 Hypothetical Labs, Inc. All Rights Reserved.
//
// This file is provided to you under the Apache License,
// Version 2.0 (the "License"); you may not use this file
// except in compliance with the License.  You may obtain
// a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.
//
// -------------------------------------------------------------------
#include <stdio.h>
#include <iostream>
#include <vector>
#include <exception>

#include "cuda.h"
#include "cuda_runtime_api.h"
#include "cublas_v2.h"
#include "erl_nif.h"

#include "numer.h"
#include "numer_buffer.h"

#include "numer_blas.h"


ErlNifResourceType *numer_buffer_resource;
ErlNifResourceType *numer_matrix_buffer_resource;
ErlNifResourceType *numer_context_resource;

ERL_NIF_TERM ATOM_TRUE;
ERL_NIF_TERM ATOM_FALSE;
ERL_NIF_TERM ATOM_OK;
ERL_NIF_TERM ATOM_ERROR;
ERL_NIF_TERM ATOM_WRONG_TYPE;
ERL_NIF_TERM OOM_ERROR;

ERL_NIF_INIT(numer_nifs, numer_nif_funcs, &numer_on_load, NULL, NULL, NULL);

extern template class NumerFloatBuffer<float>;
extern template class NumerFloatBuffer<double>;
extern template class NumerMatrixFloatBuffer<float>;
extern template class NumerMatrixFloatBuffer<double>;

static int numer_on_load(ErlNifEnv *env, void **priv_data, ERL_NIF_TERM load_info) {
    if (cuInit(0) == CUDA_SUCCESS) {
        ATOM_TRUE = enif_make_atom(env, "true");
        ATOM_FALSE = enif_make_atom(env, "false");
        ATOM_OK = enif_make_atom(env, "ok");
        ATOM_ERROR = enif_make_atom(env, "error");
        ATOM_WRONG_TYPE = enif_make_atom(env, "wrong_type");
        numer_buffer_resource = enif_open_resource_type(env, NULL, "numer_buffer_resource",
                                                            NULL, ERL_NIF_RT_CREATE, 0);
        numer_context_resource = enif_open_resource_type(env, NULL, "numer_context_resource",
                                                             NULL, ERL_NIF_RT_CREATE, 0);
        /* Pre-alloate OOM error in case we run out of memory later */
        OOM_ERROR = enif_make_tuple2(env, ATOM_ERROR, enif_make_atom(env, "out_of_memory"));
        return 0;
    }
    else {
        return -1;
    }
}

ERL_NIF_TERM numer_nifs_new_context(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    CUdevice device;
    int deviceNum = 0;    

    NumerContextRef *ref = (NumerContextRef *) enif_alloc_resource(numer_context_resource, sizeof(NumerContextRef));
    if (!ref) {
        return OOM_ERROR;
    }
    if (argc == 1 && !enif_get_int(env, argv[0], &deviceNum)) {
        return enif_make_badarg(env);
    }
    if (cuDeviceGet(&device, deviceNum) == CUDA_SUCCESS &&
        cuCtxCreate(&(ref->ctx), CU_CTX_SCHED_AUTO, device) == CUDA_SUCCESS) {
        ref->destroyed = false;
        ref->doublePrecision = false;
        ERL_NIF_TERM result = enif_make_resource(env, ref);
        enif_release_resource(ref);
        return enif_make_tuple2(env, ATOM_OK, result);
    }
    else {
        return ATOM_ERROR;
    }
}

ERL_NIF_TERM numer_nifs_destroy_context(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    NumerContextRef *ref;
    if (argc != 1 || !enif_get_resource(env, argv[0], numer_context_resource, (void **) &ref)) {
        return enif_make_badarg(env);
    }
    if (!ref->destroyed) {
        cuCtxDestroy(ref->ctx);
        ref->destroyed = true;
    }
    return ATOM_OK;
}


ERL_NIF_TERM numer_nifs_new_float_buffer(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    unsigned long size = 0;
    NumerContextRef *ctxRef;

    NumerBufferRef *ref = (NumerBufferRef *) enif_alloc_resource(numer_buffer_resource, sizeof(NumerBufferRef));
    if (!ref) {
        return OOM_ERROR;
    }

    if(argc == 0 || argc >2){
        return enif_make_badarg(env);        
    }
    else if (argc == 2 && (!enif_get_resource(env, argv[0], numer_context_resource, (void **) &ctxRef) || !enif_get_ulong(env, argv[1], &size))){ 
        return enif_make_badarg(env);
    }
    else if(argc == 1 && !enif_get_resource(env, argv[0], numer_context_resource, (void **) &ctxRef) ) {
        return enif_make_badarg(env);        
    }

    if(ctxRef->doublePrecision){
        ref->buffer = new NumerFloatBuffer<double>(size);
    }else{
        ref->buffer = new NumerFloatBuffer<float>(size);
    }  

    ref->destroyed = false;
    ERL_NIF_TERM res = enif_make_resource(env, ref);
    enif_release_resource(ref);
    return enif_make_tuple2(env, ATOM_OK, res);
}


ERL_NIF_TERM numer_nifs_destroy_buffer(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    NumerBufferRef *ref;
    if (argc != 1 || !enif_get_resource(env, argv[0], numer_buffer_resource, (void **) &ref)) {
        return enif_make_badarg(env);
    }
    if (!ref->destroyed) {
        delete ref->buffer;
        ref->destroyed = true;
    }
    return ATOM_OK;
}

ERL_NIF_TERM numer_nifs_write_buffer(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    NumerBufferRef *ref;
    ERL_NIF_TERM head;
    ERL_NIF_TERM tail;

    if (argc != 2 || !enif_get_resource(env, argv[0], numer_buffer_resource, (void **) &ref)) {
        return enif_make_badarg(env);
    }
    if (!enif_get_list_cell(env, argv[1], &head, &tail) ) {        
        return enif_make_badarg(env);
    }

    try
    {
        ref->buffer->write(env, argv[1]);
    }
    catch (std::exception& e)
    {        
        return enif_make_tuple2(env, ATOM_ERROR, enif_make_atom(env,e.what()));
    }

    return ATOM_OK;
}

ERL_NIF_TERM numer_nifs_buffer_size(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    NumerBufferRef *ref;
    if (argc != 1 || !enif_get_resource(env, argv[0], numer_buffer_resource, (void **) &ref)) {
        return enif_make_badarg(env);
    }
    return enif_make_tuple2(env, ATOM_OK, enif_make_long(env, ref->buffer->size()));
}


ERL_NIF_TERM numer_nifs_read_buffer(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    NumerBufferRef *ref;
    if (argc != 1 || !enif_get_resource(env, argv[0], numer_buffer_resource, (void **) &ref)) {
        return enif_make_badarg(env);
    }

    ERL_NIF_TERM data = ref->buffer->toErlTerms(env); 
   
    return enif_make_tuple2(env, ATOM_OK, data);
}

ERL_NIF_TERM numer_nifs_clear_buffer(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    NumerBufferRef *ref;
    if (argc != 1 || !enif_get_resource(env, argv[0], numer_buffer_resource, (void **) &ref)) {
        return enif_make_badarg(env);
    }
    ref->buffer->clear();
    return ATOM_OK;
}

//////////////////// Matrix buffer
ERL_NIF_TERM numer_nifs_new_matrix_float_buffer(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    unsigned  rows; 
    unsigned  cols;
    bool from_matrix = false;
    MatrixOrientation orientation = ROW_MAJOR;
    unsigned int mo;
    NumerContextRef *ctxRef;

    if (argc == 3) {
        ERL_NIF_TERM head;
        ERL_NIF_TERM tail;        
        if(!enif_get_resource(env, argv[0], numer_context_resource, (void **) &ctxRef) ||
           !enif_get_list_length(env, argv[1], &rows) ||
           !enif_get_list_cell(env, argv[1], &head, &tail) ||
           !enif_get_list_length(env, head, &cols) ||
           !enif_get_uint(env, argv[2], &mo))
        {            
            return enif_make_badarg(env);
        }
        from_matrix = true;
    }else if (argc !=4 || 
              !enif_get_resource(env, argv[0], numer_context_resource, (void **) &ctxRef) ||
              !enif_get_uint(env, argv[1], &rows) || 
              !enif_get_uint(env, argv[2], &cols)||
              !enif_get_uint(env, argv[3], &mo)) {
        return enif_make_badarg(env);
    }

    if(mo == ROW_MAJOR){ 
        orientation = ROW_MAJOR;
    }else if (mo == COLUMN_MAJOR){ 
        orientation = COLUMN_MAJOR;
    }else return enif_make_badarg(env);

    //NumerMatrixBufferRef *ref = (NumerMatrixBufferRef *) enif_alloc_resource(numer_buffer_resource, sizeof(NumerMatrixBufferRef));
    NumerBufferRef *ref = (NumerBufferRef *) enif_alloc_resource(numer_buffer_resource, sizeof(NumerBufferRef));
    if (!ref) {
        return OOM_ERROR;
    }

    //ref->buffer = new NumerMatrixFloatBuffer(rows, cols, orientation);
    if(ctxRef->doublePrecision){
        ref->buffer = new NumerMatrixFloatBuffer<double>(rows, cols, orientation);
    }else{
        ref->buffer = new NumerMatrixFloatBuffer<float>(rows, cols, orientation);
    }
    
    ref->destroyed = false;
    if (from_matrix) ref->buffer->write(env, argv[1]);
    

    ERL_NIF_TERM res = enif_make_resource(env, ref);
    enif_release_resource(ref);
    return enif_make_tuple2(env, ATOM_OK, res);
}


bool set_double_precision(NumerContextRef *ctxRef){
    cudaDeviceProp props;
    int deviceNum = 0;
    if(cudaGetDeviceProperties(&props, deviceNum) == CUDA_SUCCESS){
        if(props.major >= 1 && props.minor >=3){
            ctxRef->doublePrecision = true;
            return true;
        }else{
            return false;
        }
    }
    return false;
}
