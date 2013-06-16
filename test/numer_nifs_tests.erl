-module(numer_nifs_tests).

-compile(export_all).

-include("numer.hrl").
-include_lib("eunit/include/eunit.hrl").

create_destroy_float_test() ->	
    {ok, Ctx} = numer_nifs:new_context(),
    {ok, Buf} = numer_nifs:new_float_buffer(Ctx),
    ok = numer_nifs:destroy_buffer(Buf),
    numer_nifs:destroy_context(Ctx).

create_write_destroy_float_test() ->
    {ok, Ctx} = numer_nifs:new_context(),
    {ok, Buf} = numer_nifs:new_float_buffer(Ctx),
    numer_nifs:write_buffer(Buf, [0.01, 0.002, 0.0003, 0.4, 1.5]),
    {ok, 5} = numer_nifs:buffer_size(Buf),
    ok = numer_nifs:destroy_buffer(Buf),
    numer_nifs:destroy_context(Ctx).

create_destroy_float_matrix_test() ->  
    {ok, Ctx} = numer_nifs:new_context(),
    {ok, Buf} = numer_nifs:new_matrix_float_buffer(Ctx, 4,4, ?ROW_MAJOR),
    ok = numer_nifs:destroy_buffer(Buf),
    numer_nifs:destroy_context(Ctx).

create_write_destroy_matrix_float_test() ->
    {ok, Ctx} = numer_nifs:new_context(),
    {ok, Buf} = numer_nifs:new_matrix_float_buffer(Ctx, 4,4, ?ROW_MAJOR),
    A = [[16.0,2.0,3.0,13.0],[5.0,11.0,10.0,8.0],[9.0,7.0,6.0,12.0],[4.0,14.0,15.0,1.0]],
    numer_nifs:write_buffer(Buf, A),
    ok = numer_nifs:destroy_buffer(Buf),
    numer_nifs:destroy_context(Ctx).

create_write_read_destroy_matrix_float_test() ->
    {ok, Ctx} = numer_nifs:new_context(),
    {ok, Buf} = numer_nifs:new_matrix_float_buffer(Ctx, 4,4, ?ROW_MAJOR),
    A = [[16.0,2.0,3.0,13.0],[5.0,11.0,10.0,8.0],[9.0,7.0,6.0,12.0],[4.0,14.0,15.0,1.0]],
    numer_nifs:write_buffer(Buf, A),
    ?assertEqual({ok,A}, numer_nifs:read_buffer(Buf)),
    ok = numer_nifs:destroy_buffer(Buf),
    numer_nifs:destroy_context(Ctx).


create_write_read_destroy_empty_matrix_float_test() ->
    {ok, Ctx} = numer_nifs:new_context(),
    {ok, Buf} = numer_nifs:new_matrix_float_buffer(Ctx, 4,4, ?ROW_MAJOR),
    A = [[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]],
    %numer_nifs:write_buffer(Buf, A),
    ?assertEqual({ok,A}, numer_nifs:read_buffer(Buf)),
    ok = numer_nifs:destroy_buffer(Buf),
    numer_nifs:destroy_context(Ctx).
    
create_from_matrix_write_read_destroy_matrix_float_test() ->
    {ok, Ctx} = numer_nifs:new_context(),
    A = [[16.5,2.1029,3.00023,13.00001],[5.0,11.0,10.0,8.0],[9.0,7.0,6.0,12.0],[4.0,14.0,15.0,1.0]],
    {ok, Buf} = numer_nifs:new_matrix_float_buffer(Ctx, A, ?ROW_MAJOR),
    {ok, A} = numer_nifs:read_buffer(Buf),
    ok = numer_nifs:destroy_buffer(Buf),
    numer_nifs:destroy_context(Ctx).

create_matrix_float_2_test() ->
    {ok, Ctx} = numer_nifs:new_context(),
    A = [[7.0,4.0,3.0],[8.0,4.0,7.0],[15.0,6.0,99.0],[3.0,2.0,4.0]], 
    {ok, Buf} = numer_nifs:new_matrix_float_buffer(Ctx, A, ?ROW_MAJOR),
    {ok, A} = numer_nifs:read_buffer(Buf),
    ok = numer_nifs:destroy_buffer(Buf),
    numer_nifs:destroy_context(Ctx).


create_matrix_float_3_test() ->
    {ok, Ctx} = numer_nifs:new_context(),
    A = [[3.0,2.0,44.0,8.0],[5.0,7.0,12.0,21.0]], 
    {ok, Buf} = numer_nifs:new_matrix_float_buffer(Ctx, A, ?ROW_MAJOR),
    {ok, A} = numer_nifs:read_buffer(Buf),
    ok = numer_nifs:destroy_buffer(Buf),
    numer_nifs:destroy_context(Ctx).

create_float_matrix_with_int_values_test() ->
    {ok, Ctx} = numer_nifs:new_context(),
    A = [[7,4,3],[8,4,7],[15,6,99],[3,2,4]], 
    {ok, Buf} = numer_nifs:new_matrix_float_buffer(Ctx, A, ?ROW_MAJOR),
    {ok, [[7.0,4.0,3.0],[8.0,4.0,7.0],[15.0,6.0,99.0],[3.0,2.0,4.0]]} = numer_nifs:read_buffer(Buf),
    ok = numer_nifs:destroy_buffer(Buf),
    numer_nifs:destroy_context(Ctx).


create_float_matrix_with_int_values_2_test() ->
    {ok, Ctx} = numer_nifs:new_context(),
    A = [[3,2,44,8],[5,7,12,21]], 
    {ok, Buf} = numer_nifs:new_matrix_float_buffer(Ctx, A, ?ROW_MAJOR),
    {ok, [[3.0,2.0,44.0,8.0],[5.0,7.0,12.0,21.0]]} = numer_nifs:read_buffer(Buf),
    ok = numer_nifs:destroy_buffer(Buf),
    numer_nifs:destroy_context(Ctx).


negative_create_float_matrix_with_wrong_dimensions_test() ->
    {ok, Ctx} = numer_nifs:new_context(),
    {ok, Buf} = numer_nifs:new_matrix_float_buffer(Ctx, 4,3, ?ROW_MAJOR), %must be (4,4)
    A = [[16.0,2.0,3.0,13.0],[5.0,11.0,10.0,8.0],[9.0,7.0,6.0,12.0],[4.0,14.0,15.0,1.0]],
    {error,_} = numer_nifs:write_buffer(Buf, A),
    %{ok, A} = numer_nifs:read_buffer(Buf),
    ok = numer_nifs:destroy_buffer(Buf),
    numer_nifs:destroy_context(Ctx).


negative_create_float_matrix_with_wrong_dimensions_less_data_test() ->
    {ok, Ctx} = numer_nifs:new_context(),
    {ok, Buf} = numer_nifs:new_matrix_float_buffer(Ctx, 4,4, ?ROW_MAJOR), 
    A = [[16.0,2.0,3.0,13.0],[5.0,11.0,10.0,8.0],[9.0,7.0,6.0,12.0]], %one row less
    {error,_} = numer_nifs:write_buffer(Buf, A),
    %{ok, A} = numer_nifs:read_buffer(Buf),
    ok = numer_nifs:destroy_buffer(Buf),
    numer_nifs:destroy_context(Ctx).

%  
%Float matrix operations only supported
%
% GEMM: C = α op ( A ) op ( B ) + β C
gemm_test()->
    {ok, Ctx} = numer_nifs:new_context(),
    A = [[7,8,15,3],[4,4,6,2],[3,7,99,4]], %row major
    B = [[3,5],[2,7],[44,12],[8,21]], %row major
    _m = 3,%num_rows_A
    _k = 4,%num_cols_A
    _n = 2,%num_cols_B
    _alpha = 1.0,
    _beta= 0.0,
    C = [[721.0, 334.0],[300.0,162.0],[4411.0,1336.0]], %row major
    {ok, Buf_A} = numer_nifs:new_matrix_float_buffer(Ctx, A, ?ROW_MAJOR),
    {ok, Buf_B} = numer_nifs:new_matrix_float_buffer(Ctx, B, ?ROW_MAJOR),
    {ok, Buf_C} = numer_nifs:new_matrix_float_buffer(Ctx, _m,_n, ?ROW_MAJOR),
    ok = numer_nifs:gemm(Ctx, ?NO_TRANSPOSE, ?NO_TRANSPOSE, _m, _n, _k, _alpha, Buf_A, Buf_B, _beta, Buf_C),
    {ok, C} = numer_nifs:read_buffer(Buf_C),
    ok = numer_nifs:destroy_buffer(Buf_A),
    ok = numer_nifs:destroy_buffer(Buf_B),
    ok = numer_nifs:destroy_buffer(Buf_C),
    numer_nifs:destroy_context(Ctx).

negative_gemm_wrong_A_dim_test()->
    {ok, Ctx} = numer_nifs:new_context(),
    A = [[7,8,15,3],[4,4,6,2],[3,7,99,4]], %row major
    B = [[3,5],[2,7],[44,12],[8,21]], %row major
    _m = 4,%num_rows_A  WRONG!!! must be 3
    _k = 4,%num_cols_A
    _n = 2,%num_cols_B
    _alpha = 1.0,
    _beta= 0.0,
    {ok, Buf_A} = numer_nifs:new_matrix_float_buffer(Ctx, A, ?ROW_MAJOR),
    {ok, Buf_B} = numer_nifs:new_matrix_float_buffer(Ctx, B, ?ROW_MAJOR),
    {ok, Buf_C} = numer_nifs:new_matrix_float_buffer(Ctx, _m,_n, ?ROW_MAJOR),
    {error,_} = numer_nifs:gemm(Ctx, ?NO_TRANSPOSE, ?NO_TRANSPOSE, _m, _n, _k, _alpha, Buf_A, Buf_B, _beta, Buf_C),
    {ok, _} = numer_nifs:read_buffer(Buf_C),
    ok = numer_nifs:destroy_buffer(Buf_A),
    ok = numer_nifs:destroy_buffer(Buf_B),
    ok = numer_nifs:destroy_buffer(Buf_C),
    numer_nifs:destroy_context(Ctx).


%  GEMV: y <- α op ( A ) x + β y
gemv_test()->
    {ok, Ctx} = numer_nifs:new_context(),
    A = [[4.0,6.0,8.0,2.0],[5.0,7.0,9.0,3.0]],
    _m = 2, %rows A
    _n = 4, %columns A
    _alpha = 1.0,
    _beta = 0.0,
    X = [2.0,5.0,1.0,7.0],
    Y = [0.0, 0.0], 
    {ok, Buf_A} = numer_nifs:new_matrix_float_buffer(Ctx, A, ?ROW_MAJOR),
    {ok, Buf_X} = numer_nifs:new_float_buffer(Ctx),
    numer_nifs:write_buffer(Buf_X, X),
    {ok, Buf_Y} = numer_nifs:new_float_buffer(Ctx),
    numer_nifs:write_buffer(Buf_Y, Y),
    ok = numer_nifs:gemv(Ctx, ?NO_TRANSPOSE , _m, _n, _alpha, Buf_A, Buf_X, _beta, Buf_Y),
    {ok, [60.0,75.0]} = numer_nifs:read_buffer(Buf_Y),
    ok = numer_nifs:destroy_buffer(Buf_A),
    ok = numer_nifs:destroy_buffer(Buf_X),
    ok = numer_nifs:destroy_buffer(Buf_Y),
    numer_nifs:destroy_context(Ctx).

negative_gemv_wrong_A_dim_test()->
    {ok, Ctx} = numer_nifs:new_context(),
    A = [[4.0,6.0,8.0,2.0],[5.0,7.0,9.0,3.0]],
    _m = 5, %rows A  WRONG!!! must be 2
    _n = 4, %columns A
    _alpha = 1.0,
    _beta = 0.0,
    X = [2.0,5.0,1.0,7.0],
    {ok, Buf_A} = numer_nifs:new_matrix_float_buffer(Ctx, A, ?ROW_MAJOR),
    {ok, Buf_X} = numer_nifs:new_float_buffer(Ctx),
    numer_nifs:write_buffer(Buf_X, X),
    {ok, Buf_Y} = numer_nifs:new_float_buffer(Ctx),
    {error, _} = numer_nifs:gemv(Ctx, ?NO_TRANSPOSE, _m, _n, _alpha, Buf_A, Buf_X, _beta, Buf_Y),
    {ok, _} = numer_nifs:read_buffer(Buf_Y),
    ok = numer_nifs:destroy_buffer(Buf_A),
    ok = numer_nifs:destroy_buffer(Buf_X),
    ok = numer_nifs:destroy_buffer(Buf_Y),
    numer_nifs:destroy_context(Ctx).
    
%SAXPY:  y <- α * x + y
saxpy_test()->
    {ok, Ctx} = numer_nifs:new_context(),
    _a = 2.0, %!!!! this has to be float
    X = [2.0, 5.0, 1.0, 7.0],
    Y = [0.0, 0.0, 0.0, 0.0], 
    {ok, Buf_X} = numer_nifs:new_float_buffer(Ctx),
    ok = numer_nifs:write_buffer(Buf_X, X),
    {ok, Buf_Y} = numer_nifs:new_float_buffer(Ctx),
    ok = numer_nifs:write_buffer(Buf_Y, Y),
    ok = numer_nifs:saxpy(Ctx, _a, Buf_X, Buf_Y),
    {ok, [4.0, 10.0, 2.0, 14.0]} = numer_nifs:read_buffer(Buf_Y),
    ok = numer_nifs:destroy_buffer(Buf_X),
    ok = numer_nifs:destroy_buffer(Buf_Y),
    numer_nifs:destroy_context(Ctx).

negative_saxpy_sizeX_lt_sizeY_test()->
    {ok, Ctx} = numer_nifs:new_context(),
    _a = 2.0, %!!!! this has to be float
    X = [2.0, 5.0, 1.0],
    Y = [0.0, 0.0, 0.0, 0.0], 
    {ok, Buf_X} = numer_nifs:new_float_buffer(Ctx),
    ok = numer_nifs:write_buffer(Buf_X, X),
    {ok, Buf_Y} = numer_nifs:new_float_buffer(Ctx),
    ok = numer_nifs:write_buffer(Buf_Y, Y),
    {error, _} = numer_nifs:saxpy(Ctx, _a, Buf_X, Buf_Y),
    {ok, _} = numer_nifs:read_buffer(Buf_Y),
    ok = numer_nifs:destroy_buffer(Buf_X),
    ok = numer_nifs:destroy_buffer(Buf_Y),
    numer_nifs:destroy_context(Ctx).


%%%
%%% BLAS-like functions
%%%

%Transpose: B <- transpose(A)
transpose_test()->
    {ok, Ctx} = numer_nifs:new_context(),
    A = [[7,8,15,3],[4,4,6,2],[3,7,99,4]], %row major
    A_transposed = [[7.0,4.0,3.0],[8.0,4.0,7.0],[15.0,6.0,99.0],[3.0,2.0,4.0]],

    {ok, Buf_A} = numer_nifs:new_matrix_float_buffer(Ctx, A, ?ROW_MAJOR),
    {ok, Buf_B} = numer_nifs:new_matrix_float_buffer(Ctx, 4,3, ?ROW_MAJOR),
    ok = numer_nifs:transpose(Ctx, Buf_A, Buf_B),
    {ok, B} = numer_nifs:read_buffer(Buf_B),
    ?assertEqual(A_transposed, B),
    ok = numer_nifs:destroy_buffer(Buf_A),
    ok = numer_nifs:destroy_buffer(Buf_B),
    numer_nifs:destroy_context(Ctx).

% GEAM:  C = α op ( A ) + β op ( B )
% (this function is CUBLAS-specific)
geam_test()->
    {ok, Ctx} = numer_nifs:new_context(),
    A = [[7,8,15,3],[4,4,6,2],[3,7,99,4]], %row major
    B = [[1,2,3,4],[5,6,7,8],[9,10,11,12]],
    _alpha = 1.0,
    _beta = 1.0,
    _m = 3,
    _n = 4,
    {ok, Buf_A} = numer_nifs:new_matrix_float_buffer(Ctx, A, ?ROW_MAJOR),
    {ok, Buf_B} = numer_nifs:new_matrix_float_buffer(Ctx, B, ?ROW_MAJOR),
    {ok, Buf_C} = numer_nifs:new_matrix_float_buffer(Ctx, _m, _n, ?ROW_MAJOR),
    ok = numer_nifs:geam(Ctx, ?NO_TRANSPOSE, ?NO_TRANSPOSE, _m, _n, _alpha, Buf_A, _beta, Buf_B, Buf_C),
    {ok, C} = numer_nifs:read_buffer(Buf_C),
    ?assertEqual([[8.0,10.0,18.0,7.0],[9.0,10.0,13.0,10.0],[12.0,17.0,110.0,16.0]], C),
    ok = numer_nifs:destroy_buffer(Buf_A),
    ok = numer_nifs:destroy_buffer(Buf_B),
    numer_nifs:destroy_context(Ctx).

% smm (Scalar Matrix Multiply)
% B <- α * A
smm_test()->
    {ok, Ctx} = numer_nifs:new_context(),
    A = [[4.0,6.0,8.0,2.0],[5.0,7.0,9.0,3.0]],
    _m = 2, %rows A
    _n = 4, %columns A
    _alpha = 5.0,
    {ok, Buf_A} = numer_nifs:new_matrix_float_buffer(Ctx, A, ?ROW_MAJOR),
    {ok, Buf_B} = numer_nifs:new_matrix_float_buffer(Ctx, _m, _n, ?ROW_MAJOR),
    ok = numer_nifs:smm(Ctx, _alpha, Buf_A, Buf_B),
    {ok, B} = numer_nifs:read_buffer(Buf_B),
    ?assertEqual([[20.0,30.0,40.0,10.0],[25.0,35.0,45.0,15.0]], B),
    ok = numer_nifs:destroy_buffer(Buf_A),
    ok = numer_nifs:destroy_buffer(Buf_B),
    numer_nifs:destroy_context(Ctx).

% smm_vector_test()->
%     {ok, Ctx} = numer_nifs:new_context(),
%     A = [4.0,6.0,8.0,2.0,5.0,7.0,9.0,3.0],
%     _m = 2, %rows A
%     _n = 4, %columns A
%     _alpha = 5.0,
%     {ok, Buf_A} = numer_nifs:new_float_buffer(Ctx, length(A)),
%     numer_nifs:write_buffer(Buf_A, A),
%     {ok, Buf_B} = numer_nifs:new_float_buffer(Ctx, length(A)),
%     ok = numer_nifs:smm(Ctx, _alpha, Buf_A, Buf_B),
%     {ok, B} = numer_nifs:read_buffer(Buf_B),
%     ?assertEqual([20.0,30.0,40.0,10.0,25.0,35.0,45.0,15.0], B),
%     ok = numer_nifs:destroy_buffer(Buf_A),
%     ok = numer_nifs:destroy_buffer(Buf_B),
%     numer_nifs:destroy_context(Ctx).