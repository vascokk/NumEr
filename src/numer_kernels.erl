-module(numer_kernels).

-include("include/numer.hrl").

-export([sigmoid/3,
		 tanh/3,
		 log/3]).

-spec sigmoid(context(), float_vector_buffer()|float_matrix_buffer(), float_vector_buffer()|float_matrix_buffer()) -> ok. 
sigmoid(#pc_context{ref=Ctx}, #pc_buffer{ref=Buf_A}, #pc_buffer{ref=Buf_B})->
	%{ok, Ctx2} = numer_nifs:new_context(),
	numer_nifs:sigmoid(Ctx, Buf_A, Buf_B).   	

-spec tanh(context(), float_vector_buffer()|float_matrix_buffer(), float_vector_buffer()|float_matrix_buffer()) -> ok.
tanh(#pc_context{ref=Ctx}, #pc_buffer{ref=Buf_A}, #pc_buffer{ref=Buf_B})->
	numer_nifs:tanh(Ctx, Buf_A, Buf_B).


-spec log(context(), float_vector_buffer()|float_matrix_buffer(), float_vector_buffer()|float_matrix_buffer()) -> ok.
log(#pc_context{ref=Ctx}, #pc_buffer{ref=Buf_A}, #pc_buffer{ref=Buf_B})->
	numer_nifs:log(Ctx, Buf_A, Buf_B).