-module(numer_logreg_tests).
-compile(export_all).
-include("numer.hrl").
-include_lib("eunit/include/eunit.hrl").



bin_to_num(Elem) ->
    try list_to_float(Elem)
    	catch error:badarg -> list_to_integer(Elem)
    end.

readfile(FileName, EOL) ->
    {ok, Binary} = file:read_file(FileName),
    Lines = string:tokens(erlang:binary_to_list(Binary), EOL),
    [[bin_to_num(X) || X<-string:tokens(Y, ",")] || Y<-Lines].

readfile_test() ->
	Lines = readfile("../test/ex2data1.txt", "\r\n").

cost_test() ->
	TrainSet = readfile("../test/ex2data1.txt", "\r\n"),
	_m = length(TrainSet), % number of training examples
	Theta = [0,0,0], %
	X = [ lists:append([1.0], lists:sublist(TrEx, 2)) || TrEx<-TrainSet], %[1.0] - adding the bias term
	Y = lists:append( [ lists:sublist(TrEx, 3,3) || TrEx<-TrainSet] ),  
	Cost = numer_logreg:cost(Theta, X, Y),	
	?debugMsg(io_lib:format("~n Cost:~p",[Cost])),
	?assertEqual( [0.693147], Cost).

gradient_test() ->
	TrainSet = fennec_logreg:readfile("../test/ex2data1.txt", "\r\n"),
	_m = length(TrainSet), % number of training examples
	Theta = [0,0,0], %
	X = [ lists:append([1.0], lists:sublist(TrEx, 2)) || TrEx<-TrainSet], %[1.0] - adding the bias term
	Y = lists:append( [ lists:sublist(TrEx, 3,3) || TrEx<-TrainSet] ),  
	Grad = numer_logreg:gradient_descent(Theta, X, Y),	
	
	?debugMsg(io_lib:format("~n Gradient:~p",[Grad])).

learn_test_() ->
          {timeout, 60*60,
           fun() ->
                  learn()
           end}.

learn_buf_test_() ->
          {timeout, 60*60,
           fun() ->
                  learn_buf()
           end}.

learn() ->
	TrainSet = readfile("../test/ex2data1.txt", "\r\n"),
	_m = length(TrainSet), % number of training examples
	Theta = [0,0,0], %
	X = [ lists:append([1.0], lists:sublist(TrEx, 2)) || TrEx<-TrainSet], %[1.0] - adding the bias term
	Y = lists:append( [ lists:sublist(TrEx, 3,3) || TrEx<-TrainSet] ),  
	[Theta_final, J_hist] = numer_logreg:learn(Theta, X, Y, 0.01, 10, []) ,	
	?debugMsg(io_lib:format("~n Learned:~p",[[Theta_final, J_hist]])).

learn_buf() ->
	{ok, Ctx} = numer_context:new(),
	%TrainSet = numer_logreg:readfile(ex2data1.txt, "\r\n"),
	TrainSet = readfile("../test/ex2data1.txt", "\r\n"),
	_m = length(TrainSet), % number of training examples
	Theta = [0,0,0],%
	X = [ lists:append([1.0], lists:sublist(TrEx, 2)) || TrEx<-TrainSet], %[1.0] - adding the bias term
	Y = lists:append( [ lists:sublist(TrEx, 3,3) || TrEx<-TrainSet] ),  
	{ok, X_buf} = numer_buffer:new(Ctx, matrix, float, row_major, X),
	{ok, Y_buf} = numer_buffer:new(Ctx, matrix, float, row_major, [Y]), 
    {ok, T_buf} = numer_buffer:new(Ctx, matrix, float, row_major, [Theta]),
	Theta_final_buf = numer_logreg:learn_buf(Ctx, T_buf, X_buf, Y_buf, 0.001, 400, length(hd(X)), length(X)) ,	
	{ok, Theta_final} = numer_buffer:read(T_buf),
	numer_buffer:destroy(T_buf),
	numer_buffer:destroy(Y_buf),
	numer_buffer:destroy(X_buf),
	numer_context:destroy(Ctx),
	?debugMsg(io_lib:format("~n Learned:~p",[Theta_final])).
	%Learned:[-0.02788488380610943,0.010618738830089569,6.68175402097404e-4]

learn_buf2_test() ->
	{ok, Ctx} = numer_context:new(),
	%TrainSet = numer_logreg:readfile(ex2data1.txt, "\r\n"),
	TrainSet = readfile("../test/ex2data1.txt", "\r\n"),
	_m = length(TrainSet), % number of training examples
	Theta = [0,0,0],%
	X = [ lists:append([1.0], lists:sublist(TrEx, 2)) || TrEx<-TrainSet], %[1.0] - adding the bias term
	Y = lists:append( [ lists:sublist(TrEx, 3,3) || TrEx<-TrainSet] ),  
	{ok, X_buf} = numer_buffer:new(Ctx, matrix, float, row_major, X),
	{ok, Y_buf} = numer_buffer:new(Ctx, matrix, float, row_major, [Y]), 
    {ok, T_buf} = numer_buffer:new(Ctx, matrix, float, row_major, [Theta]),

    %temp buffers needed for GD
    {ok, Tmp1_buf} = numer_buffer:new(Ctx, matrix, float, row_major, 1, _m),
    {ok, H_buf} = numer_buffer:new(Ctx, matrix, float, row_major, 1, _m),
    {ok, Grad_buf} = numer_buffer:new(Ctx, matrix, float, row_major, 1, length(hd(X))),

	Theta_final_buf = numer_logreg:learn_buf(Ctx, T_buf, X_buf, Y_buf, 0.001, 400, length(hd(X)), length(X), Tmp1_buf, H_buf, Grad_buf) ,	
	{ok, Theta_final} = numer_buffer:read(T_buf),
	numer_buffer:destroy(T_buf),
	numer_buffer:destroy(Y_buf),
	numer_buffer:destroy(X_buf),
	numer_buffer:destroy(H_buf),
    numer_buffer:destroy(Tmp1_buf),
    numer_buffer:destroy(Grad_buf),
	numer_context:destroy(Ctx),
	?debugMsg(io_lib:format("~n Learned:~p",[Theta_final])).
	%Learned:[-0.02788488380610943,0.010618738830089569,6.68175402097404e-4]
	
