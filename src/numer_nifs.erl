-module(numer_nifs).

-define(NIF_API_VERSION, 2).
-define(MISSING_NIF, throw({error, missing_nif})).

-include_lib("eunit/include/eunit.hrl").
-include("include/numer.hrl").

-on_load(init/0).

-export([init/0]).

%% API
-export([new_context/0,
         new_context/1,
         destroy_context/1]).

-export([new_float_buffer/1,
         new_float_buffer/2,
         destroy_buffer/1,
         buffer_size/1]).

-export([write_buffer/2,
         read_buffer/1,
         clear_buffer/1]).

-export([new_matrix_float_buffer/3,         
         new_matrix_float_buffer/4]).

-export([gemm/11, 
         gemv/9, 
         saxpy/4, 
         transpose/3, 
         geam/10,
         smm/4,
         mulsimp/4]).

-export([sigmoid/3,
         tanh/3,
         log/3]).

%ML functions
-export([gd/6,
         gd_learn/8]).


-type transpose_op() :: ?TRANSPOSE |?NO_TRANSPOSE | ?CONJUGATE_TRANSPOSE.
-type orientation_C() :: ?ROW_MAJOR | ?COLUMN_MAJOR.

new_context() ->
    ?MISSING_NIF.

new_context(_DeviceNum) ->
    ?MISSING_NIF.

destroy_context(_Ctx) ->
    ?MISSING_NIF.

new_float_buffer(_Ctx) ->
    ?MISSING_NIF.

new_float_buffer(_Ctx, _size) ->
    ?MISSING_NIF.

destroy_buffer(_Buffer) ->
    ?MISSING_NIF.

buffer_size(_Buffer) ->
    ?MISSING_NIF.

-spec read_buffer(term()) -> {ok, int_vector() | float_vector() | int_matrix() | float_matrix()}.
read_buffer(_Buffer) ->
    ?MISSING_NIF.

-spec write_buffer(term(), int_vector() | float_vector() | int_matrix() | float_matrix()) -> ok.
write_buffer(_Buffer, _Data) ->
    ?MISSING_NIF.

clear_buffer(_Buffer) ->
    ?MISSING_NIF.

%% Matrices
-spec new_matrix_float_buffer(term(), float_matrix(), orientation_C()) -> {ok, term()}.
new_matrix_float_buffer(_Ctx, _A, _orientation) ->
    ?MISSING_NIF.   

-spec new_matrix_float_buffer(term(), matrix_rows(), matrix_columns(), orientation_C()) -> {ok, term()}.
new_matrix_float_buffer(_Ctx, _m, _n, _orientation) ->
    ?MISSING_NIF.    

-spec gemm(term(), transpose_op(), transpose_op(), matrix_rows(), matrix_columns(), matrix_rows(), float(), float_matrix(), float_matrix(), float(), float_matrix()) -> ok.
gemm(_Ctx, _transpose_op_A, _transpose_op_B, _m, _n, _k, _alpha, _A, _B, _beta, _C ) ->
    ?MISSING_NIF.

-spec gemv(term(), transpose_op(), matrix_rows(), matrix_columns(), float(), float_matrix(), float_vector(), float(), float_vector()) -> ok.
gemv(_Ctx, _transpose, _m, _n, _alpha, _A, _X, _beta, _Y) ->
    ?MISSING_NIF.

-spec saxpy(term(), float(), float_vector(), float_vector()) -> ok.
saxpy(_Ctx, _a, _X, _Y) ->
    ?MISSING_NIF.

-spec transpose(term(), float_matrix(), float_matrix()) -> ok.
transpose(_Ctx, _A, _B) ->
    ?MISSING_NIF.    

-spec geam(term(), transpose_op(), transpose_op(), matrix_rows(), matrix_columns(), integer()|float(), float_matrix(), integer()|float(), float_matrix(),  float_matrix()) -> ok.
geam(_Ctx, _transpose_op_A, _transpose_op_B, _m, _n, _alpha, _A, _beta, _B, _C ) ->
    ?MISSING_NIF.

-spec smm(term(), float(), float_matrix(), float_matrix()) -> ok.
smm(_Ctx, _alpha, _A, _B) ->
    ?MISSING_NIF.    


-spec sigmoid(term(), float_vector()|float_matrix(), float_vector()|float_matrix()) -> ok.
sigmoid(_Ctx, _A, _B) ->
    ?MISSING_NIF.

-spec tanh(term(), float_vector()|float_matrix(), float_vector()|float_matrix()) -> ok.
tanh(_Ctx, _A, _B) ->
    ?MISSING_NIF.

-spec log(term(), float_vector()|float_matrix(), float_vector()|float_matrix()) -> ok.
log(_Ctx, _A, _B) ->
    ?MISSING_NIF.

gd(_Ctx, _Theta, _X, _Y, _num_features, _num_samples) ->
    ?MISSING_NIF.

gd_learn(_Ctx, _Theta, _X, _Y, _num_features, _num_samples, _learning_rate, _iterations) ->
    ?MISSING_NIF.

-spec mulsimp(term(), float_vector()|float_matrix(), float_vector()|float_matrix(), float_vector()|float_matrix()) -> ok.
mulsimp(_Ctx, _A, _B, C) ->
    ?MISSING_NIF.    

init() ->
    PrivDir = case code:priv_dir(numer) of
                  {error, bad_name} ->
                      D = filename:dirname(code:which(?MODULE)),
                      filename:join([D, "..", "priv"]);
                  Dir ->
                      Dir
              end,
    SoName = filename:join([PrivDir, "numer_nifs"]),
    erlang:load_nif(SoName, ?NIF_API_VERSION).

