-module(numer_blas).

-include("include/numer.hrl").

-export([gemm/11, 
	     gemv/9, 
	     saxpy/4,
	     geam/10,
	     smm/4,
	     transpose/3,
	     mulsimp/4]).

-type transpose_op() :: transpose | no_transpose | conjugate_transpose.


-spec gemm(context(), transpose_op(), transpose_op(), matrix_rows(), matrix_columns(), matrix_rows(), float(), float_matrix_buffer(), float_matrix_buffer(), float(), float_matrix_buffer()) -> ok. 
gemm(#pc_context{ref=Ctx}, Transpose_A, Transpose_B, _m, _n, _k, _alpha, #pc_buffer{ref=Buf_A}, #pc_buffer{ref=Buf_B}, _beta, #pc_buffer{ref=Buf_C}) ->
	case Transpose_A of 
		transpose -> _transp_A = ?TRANSPOSE;
		no_transpose -> _transp_A = ?NO_TRANSPOSE;
		conjugate_transpose -> _transp_A = ?CONJUGATE_TRANSPOSE
	end,
	case Transpose_B of 
		transpose -> _transp_B = ?TRANSPOSE;
		no_transpose -> _transp_B = ?NO_TRANSPOSE;
		conjugate_transpose -> _transp_B = ?CONJUGATE_TRANSPOSE
	end,
	numer_nifs:gemm(Ctx, _transp_A, _transp_B, _m, _n, _k, _alpha, Buf_A, Buf_B, _beta, Buf_C).


-spec gemv(context(), transpose_op(), matrix_rows(), matrix_columns(), float(), float_matrix_buffer(), float_vector_buffer() , float(), float_vector_buffer() ) -> ok.
gemv(#pc_context{ref=Ctx}, Transpose_A , _m, _n, _alpha, #pc_buffer{ref=Buf_A}, #pc_buffer{ref=Buf_X}, _beta, #pc_buffer{ref=Buf_Y}) ->
	case Transpose_A of 
		transpose -> _transp_A = ?TRANSPOSE;
		no_transpose -> _transp_A = ?NO_TRANSPOSE;
		conjugate_transpose -> _transp_A = ?CONJUGATE_TRANSPOSE
	end,
	numer_nifs:gemv(Ctx, _transp_A , _m, _n, _alpha, Buf_A, Buf_X, _beta, Buf_Y).


-spec saxpy(context(), float(), float_vector_buffer(), float_vector_buffer() ) -> ok.
saxpy(#pc_context{ref=Ctx}, _alpha, #pc_buffer{ref=Buf_X}, #pc_buffer{ref=Buf_Y}) ->
	numer_nifs:saxpy(Ctx, _alpha, Buf_X, Buf_Y).


-spec transpose(context(), float_matrix_buffer(), float_matrix_buffer()) -> ok.
transpose(#pc_context{ref=Ctx}, #pc_buffer{ref=Buf_A}, #pc_buffer{ref=Buf_B}) ->
	numer_nifs:transpose(Ctx, Buf_A, Buf_B).    


-spec geam(context(), transpose_op(), transpose_op(), matrix_rows(), matrix_columns(), integer()|float(), float_matrix_buffer(), integer()|float(), float_matrix_buffer(),  float_matrix_buffer()) -> ok.
geam(#pc_context{ref=Ctx}, Transpose_A, Transpose_B, _m, _n, _alpha, #pc_buffer{ref=Buf_A}, _beta, #pc_buffer{ref=Buf_B}, #pc_buffer{ref=Buf_C} ) ->
	case Transpose_A of 
		transpose -> _transp_A = ?TRANSPOSE;
		no_transpose -> _transp_A = ?NO_TRANSPOSE;
		conjugate_transpose -> _transp_A = ?CONJUGATE_TRANSPOSE
	end,
	case Transpose_B of 
		transpose -> _transp_B = ?TRANSPOSE;
		no_transpose -> _transp_B = ?NO_TRANSPOSE;
		conjugate_transpose -> _transp_B = ?CONJUGATE_TRANSPOSE
	end,    
    numer_nifs:geam(Ctx, _transp_A, _transp_B, _m, _n, _alpha, Buf_A, _beta, Buf_B, Buf_C).


-spec smm(context(), float(), float_matrix_buffer(), float_matrix_buffer()) -> ok.
smm(#pc_context{ref=Ctx}, _alpha, #pc_buffer{ref=Buf_A}, #pc_buffer{ref=Buf_B}) ->
    numer_nifs:smm(Ctx, _alpha, Buf_A, Buf_B).    	

-spec mulsimp(context(), float_vector_buffer()|float_matrix_buffer(), float_vector_buffer()|float_matrix_buffer(), float_vector_buffer()|float_matrix_buffer()) -> ok.
mulsimp(#pc_context{ref=Ctx}, #pc_buffer{ref=Buf_A}, #pc_buffer{ref=Buf_B}, #pc_buffer{ref=Buf_C}) ->
	numer_nifs:mulsimp(Ctx, Buf_A, Buf_B, Buf_C).    