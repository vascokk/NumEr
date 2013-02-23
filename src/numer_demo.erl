-module(numer_demo).

-compile([export_all,
          native]).

start(N) ->
    {T1, T2, T3} = erlang:now(),
    random:seed(T1, T2, T3),
    io:format("Generating test data: ~p~n", [N]),
    D = [random:uniform(N) || _ <- lists:seq(1, N)],
    io:format("Measuring performance "),
    {Time1, _} = timer:tc(lists, sort, [D]),
    io:format("."),
    {ok, C} = numer_context:new(),
    {ok, B} = numer_buffer:new(integer),
    numer_buffer:write(B, D),
    %numer_nifs:write_buffer(B, D),
    {Time2, _} = timer:tc(numer_demo, numer_sort, [C, B, D]),
    io:format(".~n"),
    io:format("Erlang: ~pms, CUDA: ~pms~n", [Time1 / 1000, Time2 / 1000]),
    numer_buffer:destroy(B),
    numer_context:destroy(C).

numer_sort(C, B, D) ->
    numer_buffer:write(B, D),
    numer_buffer:sort(C, B),
    numer_buffer:read(B).
