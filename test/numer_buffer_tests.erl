    -module(numer_buffer_tests).

-compile(export_all).

-include_lib("eunit/include/eunit.hrl").

%%
%% Erlang Buffer API tests
%%

float_buffer_test() ->
    {ok, Ctx} = numer_context:new(),
    A = [1.01,2.02,3.03,4.04,5.05],
    {ok, Buf_A} = numer_buffer:new(Ctx, float, length(A)),
    numer_buffer:write(Buf_A, A),
    ?assertEqual({ok, [1.01,2.02,3.03,4.04,5.05]}, numer_buffer:read(Buf_A)),
    numer_context:destroy(Ctx).

float_matrix_buffer_test() ->
    {T1, T2, T3} = erlang:now(),
    random:seed(T1, T2, T3),
    Rows = random:uniform(10),
    Cols = random:uniform(10),
    M = [[random:uniform(1000)+0.001 || _ <- lists:seq(1, Cols)] || _ <- lists:seq(1, Rows)],
    {ok, Ctx} = numer_context:new(),
    {Res, Buf_M} = numer_buffer:new(Ctx, matrix, float, row_major, Rows, Cols),
    ?assertEqual(Res,ok),
    ?assertEqual(ok, numer_buffer:write(Buf_M, M)),
    ?assertEqual({ok, M}, numer_buffer:read(Buf_M)),
    ?assertEqual(ok,  numer_buffer:destroy(Buf_M)),
    ?assertEqual(ok,  numer_context:destroy(Ctx)).

float_matrix_buffer_2_test() ->
    {T1, T2, T3} = erlang:now(),
    random:seed(T1, T2, T3),
    Rows = random:uniform(1000),
    Cols = random:uniform(1000),
    M = [[random:uniform(1000)+0.001 || _ <- lists:seq(1, Cols)] || _ <- lists:seq(1, Rows)],
    {ok, Ctx} = numer_context:new(),
    {Res, Buf_M} = numer_buffer:new(Ctx, matrix, float, row_major, M), 
    ?assertEqual(Res,ok),
    ?assertEqual({ok, M}, numer_buffer:read(Buf_M)),
    ?assertEqual(ok, numer_buffer:destroy(Buf_M)),
    ?assertEqual(ok, numer_context:destroy(Ctx)).

ones_1_test() ->
    {ok, Ctx} = numer_context:new(),
    {Res, Buf_ones} = numer_buffer:ones(Ctx, float, 5),
    ?assertEqual(Res,ok),
    ?assertEqual({ok, [1.0,1.0,1.0,1.0,1.0]} , numer_buffer:read(Buf_ones)),
    ok = numer_buffer:destroy(Buf_ones),
    ok = numer_context:destroy(Ctx).

ones_2_test() ->
    {ok, Ctx} = numer_context:new(),
    {Res, Buf_ones} = numer_buffer:ones(Ctx, matrix, float, row_major, 2, 3),
    ?assertEqual(Res,ok),
    ?assertEqual({ok, [[1.0,1.0,1.0],[1.0,1.0,1.0]]} , numer_buffer:read(Buf_ones)),
    ok = numer_buffer:destroy(Buf_ones),
    ok = numer_context:destroy(Ctx).

zeros_1_test() ->
    {ok, Ctx} = numer_context:new(),
    {Res, Buf_zeros} = numer_buffer:zeros(Ctx, float, 5),
    ?assertEqual(Res,ok),
    ?assertEqual({ok, [0.0,0.0,0.0,0.0,0.0]} , numer_buffer:read(Buf_zeros)),
    ok = numer_buffer:destroy(Buf_zeros),
    ok = numer_context:destroy(Ctx).

zeros_2_test() ->
    {ok, Ctx} = numer_context:new(),
    {Res, Buf_zeros} = numer_buffer:zeros(Ctx, matrix, float, row_major, 2, 3),
    ?assertEqual(Res,ok),
    ?assertEqual({ok, [[0.0,0.0,0.0],[0.0,0.0,0.0]]} , numer_buffer:read(Buf_zeros)),
    ok = numer_buffer:destroy(Buf_zeros),
    ok = numer_context:destroy(Ctx).
