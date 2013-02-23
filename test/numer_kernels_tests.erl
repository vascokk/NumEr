-module(numer_kernels_tests).
-compile(export_all).
-include("numer.hrl").
-include_lib("eunit/include/eunit.hrl").

sigmoid_test()->
	{ok, Ctx} = numer_context:new(),
	A = [1,2,3,4,5,6,7,8,9],
	Sig = [0.73106, 0.88080, 0.95257, 0.98201, 0.99331, 0.99753, 0.99909, 0.99966, 0.99988], 
	
	{ok, Buf_A} = numer_buffer:new(Ctx, float),
	numer_buffer:write(Buf_A, A),
	
	{ok, Buf_B} = numer_buffer:new(Ctx, float, length(A)),
	
	ok = numer_kernels:sigmoid(Ctx, Buf_A, Buf_B),
	{ok, Res} = numer_buffer:read(Buf_B),
	?assertEqual(Sig, [element(1,string:to_float(hd(io_lib:format("~.5f",[X])))) || X<-Res]),
	ok = numer_buffer:destroy(Buf_A),
	ok = numer_buffer:destroy(Buf_B),
	ok = numer_context:destroy(Ctx).

tanh_test()->
	{ok, Ctx} = numer_context:new(),
	A = [1,2,3,4,5,6,7,8,9],
	Sig = [0.76159, 0.96403, 0.99505, 0.99933, 0.99991, 0.99999, 1.00000, 1.00000, 1.00000], 
	
	{ok, Buf_A} = numer_buffer:new(Ctx, float),
	numer_buffer:write(Buf_A, A),
	
	{ok, Buf_B} = numer_buffer:new(Ctx, float, length(A)),
	
	ok = numer_kernels:tanh(Ctx, Buf_A, Buf_B),
	{ok, Res} = numer_buffer:read(Buf_B),
	?assertEqual(Sig, [element(1,string:to_float(hd(io_lib:format("~.5f",[X])))) || X<-Res]),
	ok = numer_buffer:destroy(Buf_A),
	ok = numer_buffer:destroy(Buf_B),
	ok = numer_context:destroy(Ctx).

log_test()->
	{ok, Ctx} = numer_context:new(),
	A = [78.0, 105.0, 105.0, 165.0],
	Log = [4.3567, 4.6540, 4.6540, 5.1059],
	
	{ok, Buf_A} = numer_buffer:new(Ctx, float),
	numer_buffer:write(Buf_A, A),
	
	{ok, Buf_B} = numer_buffer:new(Ctx, float, length(A)),
	
	ok = numer_kernels:log(Ctx, Buf_A, Buf_B),
	{ok, Res} = numer_buffer:read(Buf_B),
	?assertEqual(Log, [element(1,string:to_float(hd(io_lib:format("~.4f",[X])))) || X<-Res]),
	ok = numer_buffer:destroy(Buf_A),
	ok = numer_buffer:destroy(Buf_B),
	ok = numer_context:destroy(Ctx).

%tanh res: 0.76159   0.96403   0.99505   0.99933   0.99991   0.99999   1.00000   1.00000   1.00000