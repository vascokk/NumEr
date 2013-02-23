-module(numer_blas_tests).

-compile(export_all).

-include("numer.hrl").

-include_lib("eunit/include/eunit.hrl").

-define(setup(F), {setup, fun start/0, fun stop/1, F}).


transpose([[]|_]) -> [];
transpose(M) ->
  [lists:map(fun hd/1, M) | transpose(lists:map(fun tl/1, M))].


transpose_gpu(Ctx, Buf_A, Buf_B) ->
  ok = numer_nifs:transpose(Ctx, Buf_A, Buf_B),
  {ok, B} = numer_nifs:read_buffer(Buf_B),
  B.

func(RowM1,M2)->
  F2 = fun(_RowM1,_RowM2) -> [X*Y || {X,Y}<-lists:zip(_RowM1,_RowM2)] end,
  F1 = fun(_RowM1,_M2) -> [lists:sum(F2(_RowM1,_RowM2)) || _RowM2<-_M2 ] end,
	F1(RowM1,M2).
	
mmul_cpu([],Acc,MR)->
		lists:reverse(Acc);
mmul_cpu([H|T],Acc, MR)->		
		L = func(H,MR),
		mmul_cpu(T,[L|Acc],MR).


benchmark_test_() ->
          {timeout, 60*60,
           fun() ->
                  mmul()
           end}.

transpose_benchmark_test_() ->
          {timeout, 60*60,
           fun() ->
                  transp_bm()
           end}.           

%% Matrix-matrix multiplication benchmark
mmul() ->
   	_m = 300,
    _k = 300,
    _n = 300,
    _alpha = 1.0,
    _beta= 0.0,
	{T1, T2, T3} = erlang:now(),
    random:seed(T1, T2, T3),
    Rows = _m,
    Cols = _k,
    M1 = [[random:uniform(1000)+0.1 || _ <- lists:seq(1, Cols)] || _ <- lists:seq(1, Rows)],
    M2 = [[random:uniform(1000)+0.1 || _ <- lists:seq(1, Cols)] || _ <- lists:seq(1, Rows)],
    
    %% CPU test
    Fun = fun(M1,M2)-> mmul_cpu(M1,[],transpose(M2)) end,
    {Time1, _} = timer:tc(Fun,[M1,M2]),
    %?debugMsg(io_lib:format("~n M result:~p",[ResM])),
   
    {ok, Ctx} = numer_nifs:new_context(),

    %% GPU CUBLAS test
    {ok, Buf_M1} = numer_nifs:new_matrix_float_buffer(Ctx, M1, ?ROW_MAJOR),
    {ok, Buf_M2} =  numer_nifs:new_matrix_float_buffer(Ctx, M2, ?ROW_MAJOR),
    {ok, Buf_C} = numer_nifs:new_matrix_float_buffer(_m,_n, ?ROW_MAJOR),
    {Time2, _} = timer:tc(numer_nifs, gemm, [Ctx, ?NO_TRANSPOSE, ?NO_TRANSPOSE, _m, _n, _k, _alpha, Buf_M1, Buf_M2, _beta, Buf_C]),


    %% CPU multiplication with GPU transpose
    {ok, Buf_M2T} = numer_nifs:new_matrix_float_buffer(_n, _k, ?ROW_MAJOR),
    {Time3, _} = timer:tc(Fun,[M1,transpose_gpu(Ctx, Buf_M1, Buf_M2T)]),   

    ok = numer_nifs:destroy_buffer(Buf_M1),
    ok = numer_nifs:destroy_buffer(Buf_M2),
    ok = numer_nifs:destroy_buffer(Buf_C),
    ok = numer_nifs:destroy_buffer(Buf_M2T),

    numer_nifs:destroy_context(Ctx),
    
    %%Print results
    ?debugMsg(io_lib:format("Execution time Erlang(CPU):~p",[Time1])),
    ?debugMsg(io_lib:format("Execution time CUDA(GPU):~p",[Time2])),
    ?debugMsg(io_lib:format("Execution time Erlang & CUDA transpose:~p",[Time3])).
    

%%
%% Erlang BLAS API tests
%%

% GEMM: C = α op ( A ) op ( B ) + β C
gemm_test()->
    {ok, Ctx} = numer_context:new(),
    A = [[7,8,15,3],[4,4,6,2],[3,7,99,4]], %row major
    B = [[3,5],[2,7],[44,12],[8,21]], %row major
    _m = 3,%num_rows_A
    _k = 4,%num_cols_A
    _n = 2,%num_cols_B
    _alpha = 1.0,
    _beta= 0.0,
    C = [[721.0, 334.0],[300.0,162.0],[4411.0,1336.0]], %row major
    {ok, Buf_A} = numer_buffer:new(Ctx, matrix, float, row_major,A),
    {ok, Buf_B} = numer_buffer:new(Ctx, matrix, float, row_major,B),
    {ok, Buf_C} = numer_buffer:new(Ctx, matrix, float, row_major,_m,_n),
    ok = numer_blas:gemm(Ctx, no_transpose, no_transpose, _m, _n, _k, _alpha, Buf_A, Buf_B, _beta, Buf_C),
    {ok, C} = numer_buffer:read(Buf_C),
    ok = numer_buffer:destroy(Buf_A),
    ok = numer_buffer:destroy(Buf_B),
    ok = numer_buffer:destroy(Buf_C),
    ok = numer_context:destroy(Ctx).

gemm_matrix_4_2_test()->
    {ok, Ctx} = numer_context:new(),
    A = [[7,8],[4,4],[3,7],[2,6]], %row major
    B = [[7,8],[4,4],[3,7],[2,6]], %row major
    _m = 2,%num_rows transpose_op(A) and C
    _k = 4,%num cols transpose_op(A) and rows transpose_op(B)
    _n = 2,%num_cols transpose_op(B) and C
    _alpha = 1.0,
    _beta= 0.0,
    C = [[78.0,105.0],[105.0,165.0]], %row major
    {ok, Buf_A} = numer_buffer:new(Ctx, matrix, float, row_major,A),
    {ok, Buf_B} = numer_buffer:new(Ctx, matrix, float, row_major,B),
    {ok, Buf_C} = numer_buffer:new(Ctx, matrix, float, row_major,_m,_n),
    ok = numer_blas:gemm(Ctx, transpose, no_transpose, _m, _n, _k, _alpha, Buf_A, Buf_B, _beta, Buf_C),
    {ok, C} = numer_buffer:read(Buf_C),
    ok = numer_buffer:destroy(Buf_A),
    ok = numer_buffer:destroy(Buf_B),
    ok = numer_buffer:destroy(Buf_C),
    ok = numer_context:destroy(Ctx).



%  GEMV: y <- α op ( A ) x + β y
gemv_test()->
    {ok, Ctx} = numer_context:new(),
    A = [[4.0,6.0,8.0,2.0],[5.0,7.0,9.0,3.0]],
    _m = 2, %rows A
    _n = 4, %columns A
    _alpha = 1.0,
    _beta = 0.0,
    X = [2.0,5.0,1.0,7.0],
    Y = [0.0, 0.0], 
    {ok, Buf_A} = numer_buffer:new(Ctx, matrix, float, row_major, A),
    {ok, Buf_X} = numer_buffer:new(Ctx, float),
    numer_buffer:write(Buf_X, X),
    {ok, Buf_Y} = numer_buffer:new(Ctx, float),
    numer_buffer:write(Buf_Y, Y),
    ok = numer_blas:gemv(Ctx, no_transpose , _m, _n, _alpha, Buf_A, Buf_X, _beta, Buf_Y),
    {ok, [60.0,75.0]} = numer_buffer:read(Buf_Y),
    ok = numer_buffer:destroy(Buf_A),
    ok = numer_buffer:destroy(Buf_X),
    ok = numer_buffer:destroy(Buf_Y),
    ok = numer_context:destroy(Ctx).

%SAXPY:  y <- a * x + y
saxpy_test()->
    {ok, Ctx} = numer_context:new(),
    _a = 2.0, %!!!! this has to be float
    X = [2.0, 5.0, 1.0, 7.0],
    Y = [0.0, 0.0, 0.0, 0.0], 
    {ok, Buf_X} = numer_buffer:new(Ctx, float),
    ok = numer_buffer:write(Buf_X, X),
    {ok, Buf_Y} = numer_buffer:new(Ctx, float),
    ok = numer_buffer:write(Buf_Y, Y),
    ok = numer_blas:saxpy(Ctx, _a, Buf_X, Buf_Y),
    {ok, [4.0, 10.0, 2.0, 14.0]} = numer_buffer:read(Buf_Y),
    ok = numer_buffer:destroy(Buf_X),
    ok = numer_buffer:destroy(Buf_Y),
    ok = numer_context:destroy(Ctx).

% GEAM:  C = α op ( A ) + β op ( B )
geam_test()->
    {ok, Ctx} = numer_context:new(),
    A = [[7,8,15,3],[4,4,6,2],[3,7,99,4]], %row major
    B = [[1,2,3,4],[5,6,7,8],[9,10,11,12]],
    _alpha = 1.0,
    _beta = 1.0,
    _m = 3,
    _n = 4,
    {ok, Buf_A} = numer_buffer:new(Ctx, matrix, float, row_major, A),
    {ok, Buf_B} = numer_buffer:new(Ctx, matrix, float, row_major, B),
    {ok, Buf_C} = numer_buffer:new(Ctx, matrix, float, row_major, _m, _n),
    ok = numer_blas:geam(Ctx, no_transpose, no_transpose, _m, _n, _alpha, Buf_A, _beta, Buf_B, Buf_C),
    {ok, C} = numer_buffer:read(Buf_C),
    ?assertEqual([[8.0,10.0,18.0,7.0],[9.0,10.0,13.0,10.0],[12.0,17.0,110.0,16.0]], C),
    ok = numer_buffer:destroy(Buf_A),
    ok = numer_buffer:destroy(Buf_B),
    ok = numer_context:destroy(Ctx).    

% B <- α * A
smm_test()->
    {ok, Ctx} = numer_context:new(),
    A = [[4.0,6.0,8.0,2.0],[5.0,7.0,9.0,3.0]],
    _m = 2, %rows A
    _n = 4, %columns A
    _alpha = 5.0,
    {ok, Buf_A} = numer_buffer:new(Ctx, matrix, float, row_major, A),
    {ok, Buf_B} = numer_buffer:new(Ctx, matrix, float, row_major, _m, _n),
    ok = numer_blas:smm(Ctx, _alpha, Buf_A, Buf_B),
    {ok, B} = numer_buffer:read(Buf_B),
    ?assertEqual([[20.0,30.0,40.0,10.0],[25.0,35.0,45.0,15.0]], B),
    ok = numer_buffer:destroy(Buf_A),
    ok = numer_buffer:destroy(Buf_B),
    ok = numer_context:destroy(Ctx).

transpose_test()->
    Rows = 500,
    Cols = 500,
    {T1, T2, T3} = erlang:now(),
    random:seed(T1, T2, T3),
    M = [[random:uniform(1000)+0.1 || _ <- lists:seq(1, Cols)] || _ <- lists:seq(1, Rows)],
    {ok, Ctx} = numer_context:new(),

    {ok, Buf_M} = numer_buffer:new(Ctx, matrix, float, row_major, M),
    {ok, Buf_MT} =  numer_buffer:new(Ctx, matrix, float, row_major, Cols, Rows),
    ok = numer_blas:transpose(Ctx, Buf_M, Buf_MT),
    ok = numer_buffer:destroy(Buf_M),
    ok = numer_buffer:destroy(Buf_MT),
    ok = numer_context:destroy(Ctx).


transp_bm()->
    Rows = 2000,
    Cols = 2000,
    {T1, T2, T3} = erlang:now(),
    random:seed(T1, T2, T3),
    M = [[random:uniform(1000)+0.1 || _ <- lists:seq(1, Cols)] || _ <- lists:seq(1, Rows)],
    {ok, Ctx} = numer_context:new(),

    {ok, Buf_M} = numer_buffer:new(Ctx, matrix, float, row_major, M),
    {ok, Buf_MT} =  numer_buffer:new(Ctx, matrix, float, row_major, Cols, Rows),
    Fun = fun(M1) -> transpose(M1) end,
    {Time1, _} = timer:tc(numer_blas, transpose, [Ctx, Buf_M, Buf_MT]),
    {Time2, _} = timer:tc(Fun, [M]),
    ok = numer_buffer:destroy(Buf_M),
    ok = numer_buffer:destroy(Buf_MT),
    ok = numer_context:destroy(Ctx),

    ?debugMsg(io_lib:format("Transpose GPU:~p",[Time1])),
    ?debugMsg(io_lib:format("Transpose CPU:~p",[Time2])).

mulsimp_test() ->
    A = [[7,8,15,3],[4,4,6,2],[3,7,99,4]], %row major
    B = [[1,2,3,4],[5,6,7,8],[9,10,11,12]],
    Res = [[7.0, 16.0, 45.0, 12.0],[20.0, 24.0, 42.0, 16.0],[27.0, 70.0, 1089.0, 48.0]],
    {ok, Ctx} = numer_context:new(),
    {ok, Buf_A} = numer_buffer:new(Ctx, matrix, float, row_major, A),
    {ok, Buf_B} = numer_buffer:new(Ctx, matrix, float, row_major, B),
    {ok, Buf_C} = numer_buffer:new(Ctx, matrix, float, row_major, length(A), length(hd(A))),
    ok = numer_blas:mulsimp(Ctx, Buf_A, Buf_B, Buf_C),
    {ok, C} = numer_buffer:read(Buf_C),
    ?assertEqual(Res, C),
    ok = numer_buffer:destroy(Buf_A),
    ok = numer_buffer:destroy(Buf_B),
    ok = numer_buffer:destroy(Buf_C),
    ok = numer_context:destroy(Ctx).
