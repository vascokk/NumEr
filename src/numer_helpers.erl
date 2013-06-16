-module(numer_helpers).

-include("include/numer.hrl").

%% BLAS helpers
-export([sum_by_cols/1, 
	     gemv/6, 
	     saxpy/3,
	     gemm/7,
	     smm/2,
	     m2v/1,
	     v2m/2,
	     transpose/1]).

%% Kernels helpers
-export([sigmoid/1,
	     tanh/1,
	     log/1]).

-export([zeros/1,
		 zeros/2,
		 ones/1,
		 ones/2]).

-type transpose_op() :: transpose | no_transpose | conjugate_transpose.


-spec sum_by_cols(float_matrix()) -> float_vector(). 
sum_by_cols(Matrix) ->
	{ok, Ctx} = numer_context:new(),
    _m = length(Matrix), %rows A
    _n = length(hd(Matrix)), %columns A
    _alpha = 1.0,
    _beta = 0.0,
    {ok, Ones} = numer_buffer:ones(Ctx, float, _m),  
    {ok, Buf_M} = numer_buffer:new(Ctx, matrix, float, row_major, Matrix),
    {ok, Buf_Sum} = numer_buffer:new(Ctx, float, _m),
    ok = numer_blas:gemv(Ctx, no_transpose , _m, _n, _alpha, Buf_M, Ones, _beta, Buf_Sum),
    {ok, Res} = numer_buffer:read(Buf_Sum),
    ok = numer_buffer:destroy(Buf_M),
    ok = numer_buffer:destroy(Ones),
    ok = numer_buffer:destroy(Buf_Sum),
    ok = numer_context:destroy(Ctx),
    Res.

-spec gemv(transpose_op(), float(), float_matrix(), float_vector(), float(), float_vector()) -> float_vector().
gemv(_transpose_A, _alpha, A, X, _beta, Y) when is_number(hd(A)) ->
	gemv(_transpose_A, _alpha, [A], X, _beta, Y);
gemv(_transpose_A, _alpha, A, X, _beta, Y) when is_list(hd(A)) ->
	{ok, Ctx} = numer_context:new(),
    %_m = length(A), %rows A
    %_n = length(hd(A)), %columns A
    case _transpose_A of 
    	no_transpose -> _m = length(A), %num_rows transpose_op(A) 
    		  		    _n = length(hd(A));%num cols transpose_op(A)
    		  	   _->  _m = length(hd(A)),
    		  		    _n = length(A)
    end,
    {ok, Buf_A} = numer_buffer:new(Ctx, matrix, float, row_major, A),
    {ok, Buf_X} = numer_buffer:new(Ctx, float), 
    numer_buffer:write(Buf_X, X), 
    case Y of
    	[] -> {ok, Buf_Y} = numer_buffer:zeros(Ctx, float, _m);
    	 _ -> {ok, Buf_Y} = numer_buffer:new(Ctx, float),
    	 	  numer_buffer:write(Buf_Y, Y)		
    end, 
    ok = numer_blas:gemv(Ctx, _transpose_A , _m, _n, _alpha, Buf_A, Buf_X, _beta, Buf_Y),
    {ok, Res} = numer_buffer:read(Buf_Y),
    ok = numer_buffer:destroy(Buf_A),
    ok = numer_buffer:destroy(Buf_X),
    ok = numer_buffer:destroy(Buf_Y),
    ok = numer_context:destroy(Ctx),
    Res.

-spec saxpy(float(), float_vector(), float_vector() ) -> float_vector().
saxpy(_a, X, Y) ->
    {ok, Ctx} = numer_context:new(), 
    {ok, Buf_X} = numer_buffer:new(Ctx, float),
    ok = numer_buffer:write(Buf_X, X),
    {ok, Buf_Y} = numer_buffer:new(Ctx, float),
    ok = numer_buffer:write(Buf_Y, Y),
    ok = numer_blas:saxpy(Ctx, _a, Buf_X, Buf_Y),
    {ok, Res} = numer_buffer:read(Buf_Y),
    ok = numer_buffer:destroy(Buf_X),
    ok = numer_buffer:destroy(Buf_Y),
    ok = numer_context:destroy(Ctx),
    Res.

-spec gemm(transpose_op(), transpose_op(), float(), float_matrix(), float_matrix(), float(), float_matrix()) -> float_matrix().	 
gemm(_transpose_A, _transpose_B, _alpha, A, B, _beta, C) ->
    {ok, Ctx} = numer_context:new(),
    case _transpose_A of 
    	no_transpose -> _m = length(A), %num_rows transpose_op(A) and C
    		  		    _k = length(hd(A));%num cols transpose_op(A) and rows transpose_op(B)
    		  	   _->  _m = length(hd(A)),
    		  		    _k = length(A)
    end,
	case _transpose_B of 
    	no_transpose -> _n = length(hd(B));%num_cols transpose_op(B) and C
    		  	   _->  _n = length(B)
    end,
    {ok, Buf_A} = numer_buffer:new(Ctx, matrix, float, row_major,A),
    {ok, Buf_B} = numer_buffer:new(Ctx, matrix, float, row_major,B),
    case C of
    	[] -> {ok, Buf_C} = numer_buffer:zeros(Ctx, matrix, float, row_major, _m, _n);
    	 _ -> {ok, Buf_C} = numer_buffer:new(Ctx, matrix, float, row_major, C)
    end,    
    ok = numer_blas:gemm(Ctx, _transpose_A, _transpose_B, _m, _n, _k, _alpha, Buf_A, Buf_B, _beta, Buf_C),
    {ok, Res} = numer_buffer:read(Buf_C),
    ok = numer_buffer:destroy(Buf_A),
    ok = numer_buffer:destroy(Buf_B),
    ok = numer_buffer:destroy(Buf_C),
    ok = numer_context:destroy(Ctx),
    Res.

smm(_alpha, A)->
    {ok, Ctx} = numer_context:new(),
    _m = length(A), %rows A
    _n = length(hd(A)), %columns A
    {ok, Buf_A} = numer_buffer:new(Ctx, matrix, float, row_major, A),
    {ok, Buf_B} = numer_buffer:new(Ctx, matrix, float, row_major, _m, _n),
    ok = numer_blas:smm(Ctx, _alpha, Buf_A, Buf_B),
    {ok, Res} = numer_buffer:read(Buf_B),
    ok = numer_buffer:destroy(Buf_A),
    ok = numer_buffer:destroy(Buf_B),
    ok = numer_context:destroy(Ctx),
    Res.

-spec m2v(float_matrix()) -> float_vector().
m2v(Matrix) ->
	lists:append(Matrix).

-spec v2m(float_vector(), integer()) -> float_matrix().
v2m(List, Cols) ->
	if length(List) rem Cols /= 0 -> erlang:error(badarg);	
		true -> lists:foldr(fun(E, []) -> [[E]]; 
	                 (E, [H|RAcc]) when length(H) < Cols -> [[E|H]|RAcc] ;
	                 (E, [H|RAcc]) -> [[E],H|RAcc]
	              end, [], List)
	end.

-spec transpose(float_matrix()) -> float_matrix().
transpose(Matrix) ->
    Rows = length(Matrix),
    Cols = length(hd(Matrix)),
    {ok, Ctx} = numer_context:new(),
    {ok, Buf_M} = numer_buffer:new(Ctx, matrix, float, row_major, Matrix),
    {ok, Buf_MT} =  numer_buffer:new(Ctx, matrix, float, row_major, Cols, Rows),
    ok = numer_blas:transpose(Ctx, Buf_M, Buf_MT),
    {ok, Res} = numer_buffer:read(Buf_MT),
    ok = numer_buffer:destroy(Buf_M),
    ok = numer_buffer:destroy(Buf_MT),
    ok = numer_context:destroy(Ctx),
    Res.

-spec sigmoid(float_matrix()) -> float_matrix();
             (float_vector()) -> float_vector().
sigmoid(A) when is_list(hd(A)) ->
	{ok, Ctx} = numer_context:new(),
	{ok, Buf_A} = numer_buffer:new(Ctx, matrix, float, row_major, A),
	{ok, Buf_B} = numer_buffer:new(Ctx, matrix, float, row_major, length(A), length(hd(A))),
	
	ok = numer_kernels:sigmoid(Ctx, Buf_A, Buf_B),
	{ok, Res} = numer_buffer:read(Buf_B),
	ok = numer_buffer:destroy(Buf_A),
	ok = numer_buffer:destroy(Buf_B),
	ok = numer_context:destroy(Ctx),
	Res;
sigmoid(A) when is_number(hd(A)) ->
	{ok, Ctx} = numer_context:new(),
	{ok, Buf_A} = numer_buffer:new(Ctx, float),
	numer_buffer:write(Buf_A, A),
	{ok, Buf_B} = numer_buffer:new(Ctx, float, length(A)),
	ok = numer_kernels:sigmoid(Ctx, Buf_A, Buf_B),
	{ok, Res} = numer_buffer:read(Buf_B),
	ok = numer_buffer:destroy(Buf_A),
	ok = numer_buffer:destroy(Buf_B),
	ok = numer_context:destroy(Ctx),
	Res.

-spec tanh(float_matrix()) -> float_matrix();
             (float_vector()) -> float_vector().
tanh(A) when is_list(hd(A)) ->
	{ok, Ctx} = numer_context:new(),
	{ok, Buf_A} = numer_buffer:new(Ctx, matrix, float, row_major, A),
	{ok, Buf_B} = numer_buffer:new(Ctx, matrix, float, row_major, length(A), length(hd(A))),
	
	ok = numer_kernels:tanh(Ctx, Buf_A, Buf_B),
	{ok, Res} = numer_buffer:read(Buf_B),
	ok = numer_buffer:destroy(Buf_A),
	ok = numer_buffer:destroy(Buf_B),
	ok = numer_context:destroy(Ctx),
	Res;
tanh(A) when is_number(hd(A)) ->
	{ok, Ctx} = numer_context:new(),
	{ok, Buf_A} = numer_buffer:new(Ctx, float),
	numer_buffer:write(Buf_A, A),
	{ok, Buf_B} = numer_buffer:new(Ctx, float, length(A)),
	ok = numer_kernels:tanh(Ctx, Buf_A, Buf_B),
	{ok, Res} = numer_buffer:read(Buf_B),
	ok = numer_buffer:destroy(Buf_A),
	ok = numer_buffer:destroy(Buf_B),
	ok = numer_context:destroy(Ctx),
	Res.

-spec log(float_matrix()) -> float_matrix();
         (float_vector()) -> float_vector().
log(A) when is_list(hd(A)) ->
	{ok, Ctx} = numer_context:new(),
	{ok, Buf_A} = numer_buffer:new(Ctx, matrix, float, row_major, A),
	{ok, Buf_B} = numer_buffer:new(Ctx, matrix, float, row_major, length(A), length(hd(A))),
	
	ok = numer_kernels:log(Ctx, Buf_A, Buf_B),
	{ok, Res} = numer_buffer:read(Buf_B),
	ok = numer_buffer:destroy(Buf_A),
	ok = numer_buffer:destroy(Buf_B),
	ok = numer_context:destroy(Ctx),
	Res;
log(A) when is_number(hd(A)) ->
	{ok, Ctx} = numer_context:new(),
	{ok, Buf_A} = numer_buffer:new(Ctx, float),
	numer_buffer:write(Buf_A, A),
	{ok, Buf_B} = numer_buffer:new(Ctx, float, length(A)),
	ok = numer_kernels:log(Ctx, Buf_A, Buf_B),
	{ok, Res} = numer_buffer:read(Buf_B),
	ok = numer_buffer:destroy(Buf_A),
	ok = numer_buffer:destroy(Buf_B),
	ok = numer_context:destroy(Ctx),
	Res.

-spec ones(integer()) -> float_vector().
ones(Size) ->
    [1.0 || X<-lists:seq(1, Size)].

-spec ones(matrix_rows(), matrix_columns()) -> float_matrix().
ones(Rows, Cols) ->
    [[1.0 || _ <- lists:seq(1, Cols)] || _ <- lists:seq(1, Rows)].

-spec zeros(integer()) -> buffer().
zeros(Size) ->
    [0.0 || X<-lists:seq(1, Size)].

-spec zeros(matrix_rows(), matrix_columns()) -> buffer().
zeros(Rows, Cols) ->
    [[0.0 || _ <- lists:seq(1, Cols)] || _ <- lists:seq(1, Rows)].