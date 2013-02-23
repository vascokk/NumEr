-module(numer_helpers_tests).

-compile(export_all).

-include("numer.hrl").

-include_lib("eunit/include/eunit.hrl").


gemm_test()->
    A = [[7,8],[4,4],[3,7],[2,6]], %row major
    B = [[7,8],[4,4],[3,7],[2,6]], %row major
    _alpha = 1.0,
    _beta= 0.0,
    C = [[0.0,0.0],[0.0,0.0]], %row major
    Res = numer_helpers:gemm(transpose, no_transpose, _alpha, A, B, _beta, C),
    ?assertEqual([[78.0,105.0],[105.0,165.0]], Res).

gemm_2_test()->
    A = [[7,8],[4,4],[3,7],[2,6]], %row major
    B = [[7,8],[4,4],[3,7],[2,6]], %row major
    _alpha = 1.0,
    _beta= 0.0,
    Res = numer_helpers:gemm(transpose, no_transpose, _alpha, A, B, _beta, []),
    ?assertEqual([[78.0,105.0],[105.0,165.0]], Res).

sum_by_cols_test()->
	A = [[7, 8],[4, 4],[3, 7],[2, 6]],
	Res = numer_helpers:sum_by_cols(A),
	?assertEqual([15.0, 8.0, 10.0, 8.0], Res).

gemv_test()->
    A = [[4.0,6.0,8.0,2.0],[5.0,7.0,9.0,3.0]],
    _alpha = 1.0,
    _beta = 0.0,
    X = [2.0,5.0,1.0,7.0],
    Y = [0.0, 0.0], 
    Res = numer_helpers:gemv(no_transpose , _alpha, A, X, _beta, Y),
    ?assertEqual([60.0,75.0], Res).

gemv_2_test()->
    A = [[4.0,6.0,8.0,2.0],[5.0,7.0,9.0,3.0]],
    _alpha = 1.0,
    _beta = 0.0,
    X = [2.0,5.0,1.0,7.0],
    Res = numer_helpers:gemv(no_transpose , _alpha, A, X, _beta, []),
    ?assertEqual([60.0,75.0], Res).

gemv_3_test()->
    A = [1,2,3,4,5],
    B = [2,2,2,2,2],
    _alpha = 1.0,
    _beta = 0.0,
    Res = numer_helpers:gemv(no_transpose, _alpha, A, B, _beta, []),
    ?assertEqual([30.0], Res).

 saxpy_test()->
    _a = 2.0, %!!!! this has to be float
    X = [2.0, 5.0, 1.0, 7.0],
    Y = [0.0, 0.0, 0.0, 0.0], 
    Res = numer_helpers:saxpy(_a, X, Y),
    ?assertEqual([4.0, 10.0, 2.0, 14.0], Res).

smm_test()->
    A = [[4.0,6.0,8.0,2.0],[5.0,7.0,9.0,3.0]],
    _alpha = 5.0,
    Res = numer_helpers:smm(_alpha, A),
    ?assertEqual([[20.0,30.0,40.0,10.0],[25.0,35.0,45.0,15.0]], Res).


m2v_test() ->
	A = [[7,8],[4,4],[3,7],[2,6]],
	Res = numer_helpers:m2v(A),
	?assertEqual([7,8,4,4,3,7,2,6], Res).

v2m_test() ->
	A = [7,8,4,4,3,7,2,6],
	Res = numer_helpers:v2m(A, 2),
	?assertEqual([[7,8],[4,4],[3,7],[2,6]], Res).

transpose_test()->
	A = [[1,1,1],[2,2,2],[3,3,3],[4,4,4]],
    Res = numer_helpers:transpose(A),
    ?assertEqual([[1.0, 2.0, 3.0, 4.0],[1.0, 2.0, 3.0, 4.0],[1.0, 2.0, 3.0, 4.0]], Res).

sigmoid_test()->
	A = [1,2,3,4,5,6,7,8,9],
	Sig = [0.73106, 0.88080, 0.95257, 0.98201, 0.99331, 0.99753, 0.99909, 0.99966, 0.99988], 
	Res = numer_helpers:sigmoid(A),
	?assertEqual(Sig, [element(1,string:to_float(hd(io_lib:format("~.5f",[X])))) || X<-Res]).

sigmoid_2_test()->
	A = [[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9]],
	Sig = [[0.73106, 0.88080, 0.95257, 0.98201, 0.99331, 0.99753, 0.99909, 0.99966, 0.99988],[0.73106, 0.88080, 0.95257, 0.98201, 0.99331, 0.99753, 0.99909, 0.99966, 0.99988]], 
	Res = numer_helpers:sigmoid(A),
	?assertEqual(Sig, [[element(1,string:to_float(hd(io_lib:format("~.5f",[X])))) || X<-M] || M<-Res]).

tanh_test()->
	A = [1,2,3,4,5,6,7,8,9],
	Sig = [0.76159, 0.96403, 0.99505, 0.99933, 0.99991, 0.99999, 1.00000, 1.00000, 1.00000], 
	Res = numer_helpers:tanh(A),
	?assertEqual(Sig, [element(1,string:to_float(hd(io_lib:format("~.5f",[X])))) || X<-Res]).

tanh_2_test()->
	A = [[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9]],
	Sig = [[0.76159, 0.96403, 0.99505, 0.99933, 0.99991, 0.99999, 1.00000, 1.00000, 1.00000],[0.76159, 0.96403, 0.99505, 0.99933, 0.99991, 0.99999, 1.00000, 1.00000, 1.00000]], 
	Res = numer_helpers:tanh(A),
	?assertEqual(Sig, [[element(1,string:to_float(hd(io_lib:format("~.5f",[X])))) || X<-M] || M<-Res]).	

log_test()->
	A = [78.0, 105.0, 105.0, 165.0],
	Sig = [4.3567, 4.6540, 4.6540, 5.1059], 
	Res = numer_helpers:log(A),
	?assertEqual(Sig, [element(1,string:to_float(hd(io_lib:format("~.4f",[X])))) || X<-Res]).

log_2_test()->
	A = [[78.0, 105.0, 105.0, 165.0],[78.0, 105.0, 105.0, 165.0]],
	Sig = [[4.3567, 4.6540, 4.6540, 5.1059],[4.3567, 4.6540, 4.6540, 5.1059]], 
	Res = numer_helpers:log(A),
	?assertEqual(Sig, [[element(1,string:to_float(hd(io_lib:format("~.4f",[X])))) || X<-M] || M<-Res]).	

ones_test() ->
	?assertEqual(numer_helpers:ones(5), [1.0, 1.0, 1.0, 1.0, 1.0]).
ones_2_test() ->
	?assertEqual(numer_helpers:ones(2,2), [[1.0, 1.0], [1.0, 1.0]]).
zeros_test() ->
	?assertEqual(numer_helpers:zeros(5), [0.0, 0.0, 0.0, 0.0, 0.0]).
zeros_2_test() ->
	?assertEqual(numer_helpers:zeros(2,2), [[0.0, 0.0], [0.0, 0.0]]).