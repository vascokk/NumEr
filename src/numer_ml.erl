-module(numer_ml).

-include("include/numer.hrl").

-export([gd/6,
		 gd_learn/8]).

-spec gd(context(), float_vector_buffer(), float_matrix_buffer(), float_vector_buffer(), integer(), integer() ) -> ok.
gd(#pc_context{ref=Ctx}, #pc_buffer{ref=Buf_Theta}, #pc_buffer{ref=Buf_X}, #pc_buffer{ref=Buf_Y},_num_features, _num_samples) ->
	numer_nifs:gd(Ctx, Buf_Theta, Buf_X, Buf_Y, _num_features, _num_samples).

-spec gd_learn(context(), float_vector_buffer(), float_matrix_buffer(), float_vector_buffer(), integer(), integer(), float(), integer() ) -> ok.
gd_learn(#pc_context{ref=Ctx}, #pc_buffer{ref=Buf_Theta}, #pc_buffer{ref=Buf_X}, #pc_buffer{ref=Buf_Y},_num_features, _num_samples, _learning_rate, _iterations) ->
	numer_nifs:gd_learn(Ctx, Buf_Theta, Buf_X, Buf_Y, _num_features, _num_samples, _learning_rate, _iterations).