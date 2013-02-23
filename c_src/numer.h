#ifndef numer
#define numer

#include "erl_nif.h"
#include "numer_buffer.h"
#include "cuda.h"
#include "numer_blas.h"


extern "C" {
    static int numer_on_load(ErlNifEnv *env, void **priv_data, ERL_NIF_TERM load_info);

    ERL_NIF_TERM numer_nifs_new_context(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
    ERL_NIF_TERM numer_nifs_destroy_context(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);

    ERL_NIF_TERM numer_nifs_new_float_buffer(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);

    ERL_NIF_TERM numer_nifs_destroy_buffer(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
    ERL_NIF_TERM numer_nifs_buffer_size(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);

    ERL_NIF_TERM numer_nifs_write_buffer(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
    ERL_NIF_TERM numer_nifs_read_buffer(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
    ERL_NIF_TERM numer_nifs_buffer_delete(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
    ERL_NIF_TERM numer_nifs_buffer_insert(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
    ERL_NIF_TERM numer_nifs_clear_buffer(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
    ERL_NIF_TERM numer_nifs_buffer_contains(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
    ERL_NIF_TERM numer_nifs_copy_buffer(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);

    ERL_NIF_TERM numer_nifs_new_matrix_float_buffer(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
    
    ERL_NIF_TERM numer_nifs_gemm(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
    ERL_NIF_TERM numer_nifs_gemv(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
    ERL_NIF_TERM numer_nifs_saxpy(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
    ERL_NIF_TERM numer_nifs_transpose(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
    ERL_NIF_TERM numer_nifs_geam(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
    ERL_NIF_TERM numer_nifs_smm(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
    ERL_NIF_TERM numer_nifs_mulsimp(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
    
    ERL_NIF_TERM numer_nifs_sigmoid(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
    ERL_NIF_TERM numer_nifs_tanh(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
    ERL_NIF_TERM numer_nifs_log(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);

    //ML functions
    ERL_NIF_TERM numer_ml_gd(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
    ERL_NIF_TERM numer_ml_gd_learn(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);


    static ErlNifFunc numer_nif_funcs[] = {
        {"new_context", 0, numer_nifs_new_context},
        {"new_context", 1, numer_nifs_new_context},
        {"destroy_context", 1, numer_nifs_destroy_context},        
        {"new_float_buffer", 1, numer_nifs_new_float_buffer},
        {"new_float_buffer", 2, numer_nifs_new_float_buffer},
        {"destroy_buffer", 1, numer_nifs_destroy_buffer},
        {"buffer_size", 1, numer_nifs_buffer_size},
        {"write_buffer", 2, numer_nifs_write_buffer},
        {"read_buffer", 1, numer_nifs_read_buffer},
        {"clear_buffer", 1, numer_nifs_clear_buffer},


        {"new_matrix_float_buffer", 3, numer_nifs_new_matrix_float_buffer},
        {"new_matrix_float_buffer", 4, numer_nifs_new_matrix_float_buffer},

        {"gemm", 11, numer_nifs_gemm},
        {"gemv", 9, numer_nifs_gemv},
        {"saxpy", 4, numer_nifs_saxpy},
        {"transpose", 3, numer_nifs_transpose},
        {"geam", 10, numer_nifs_geam},
        {"smm", 4, numer_nifs_smm},
        {"sigmoid", 3, numer_nifs_sigmoid},
        {"tanh", 3, numer_nifs_tanh},
        {"log", 3, numer_nifs_log},
        {"mulsimp", 4, numer_nifs_mulsimp},
        
        //ML functions
        {"gd", 6, numer_ml_gd},
        {"gd_learn", 8, numer_ml_gd_learn}      

    };
};

struct NumerBufferRef {
    NumerBuffer *buffer;
    bool destroyed;
};

struct NumerMatrixBufferRef {
    NumerMatrixBuffer *buffer;
    bool destroyed;
};

struct NumerContextRef {
    CUcontext ctx;
    bool destroyed;
    bool doublePrecision;
};

extern ErlNifResourceType *numer_buffer_resource;
extern ErlNifResourceType *numer_context_resource;

extern ERL_NIF_TERM ATOM_TRUE;
extern ERL_NIF_TERM ATOM_FALSE;
extern ERL_NIF_TERM ATOM_OK;
extern ERL_NIF_TERM ATOM_ERROR;
extern ERL_NIF_TERM ATOM_WRONG_TYPE;
extern ERL_NIF_TERM OOM_ERROR;


#endif
