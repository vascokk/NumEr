-module(numer_stress).

-export([run/0]).

run() ->
    {T1, T2, T3} = erlang:now(),
    random:seed(T1, T2, T3),
    F = fun(_, _) -> random:uniform(100) > 50 end,
    Data = lists:sort(F, lists:seq(1, 1000000)),
    io:format("Pid: ~p~n", [os:getpid()]),
    io:get_chars("Press any key when ready...", 1),
    {ok, Ctx} = numer_nifs:new_context(),
    stress(Ctx, Data, 1000000).

stress(_, _Data, 0) ->
    ok;
stress(Ctx, Data, Count) ->
    {ok, B} = numer_nifs:new_int_buffer(),    
    numer_nifs:write_buffer(B, Data),
    %io:format("~p~n", [B]),
    numer_nifs:sort_buffer(Ctx,B),
    %{ok, SD} = numer_nifs:read_buffer(B),
    numer_nifs:destroy_buffer(B),
    io:format("~p~n", [1000000 - Count]),
    %% case length(SD) of
    %%     1000000 ->
    %%         io:format("~p...ok~n", [1000000 - Count]);
    %%     _ ->
    %%         io:format("~p...bad~n", [1000000 - Count])
    %% end,    
    stress(Ctx, Data, Count - 1).
