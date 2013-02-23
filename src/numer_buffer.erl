-module(numer_buffer).

-include("include/numer.hrl").

-export([new/2, new/3, new/5, new/6,
         destroy/1,
         size/1,
         write/2,
         read/1,
         clear/1,
         ones/3,
         ones/6,
         zeros/3,
         zeros/6]).

-spec new(term(), data_type) -> {ok, buffer()}.
new(#pc_context{ref=Ctx}, float) ->
    {ok, Buf} = numer_nifs:new_float_buffer(Ctx),
    {ok, #pc_buffer{type = vector, data_type=float, ref=Buf}}.

-spec new(term(), data_type, integer()) -> {ok, buffer()}.
new(#pc_context{ref=Ctx}, float, Size) ->
    {ok, Buf} = numer_nifs:new_float_buffer(Ctx, Size),
    %numer_nifs:write_buffer(Buf, [0.0|| X<-lists:seq(1,Size)]),
    {ok, #pc_buffer{type = vector, data_type=float, ref=Buf}}.


-spec new(term(), matrix, data_type(), orientation(), matrix_rows(), matrix_columns()) -> {ok, buffer()}.
new(#pc_context{ref=Ctx}, matrix, float, Orientation, Rows, Cols) ->
    case Orientation of 
        row_major -> _orientation = ?ROW_MAJOR;
        column_major -> _orientation = ?COLUMN_MAJOR
    end,
    {ok, Buf} = numer_nifs:new_matrix_float_buffer(Ctx, Rows,Cols, _orientation),
    {ok, #pc_buffer{type = matrix, data_type=float, orientation=Orientation, ref=Buf}}.

-spec new(term(), matrix, float, orientation(), float_matrix()) -> {ok, buffer()}.
new(#pc_context{ref=Ctx}, matrix, float, Orientation, Matrix) ->
    case Orientation of 
        row_major -> _orientation = ?ROW_MAJOR;
        column_major -> _orientation = ?COLUMN_MAJOR
    end,
    {ok, Buf} = numer_nifs:new_matrix_float_buffer(Ctx, Matrix, _orientation),
    {ok, #pc_buffer{type = matrix, data_type=float, orientation=Orientation,  ref=Buf}}.    

-spec ones(term(), data_type, integer()) -> {ok, buffer()}.
ones(#pc_context{ref=Ctx}, float, Size) ->
    {ok, Buf} = new(Ctx, float),
    ok = write(Buf, [1.0 || X<-lists:seq(1, Size)]),
    {ok, Buf}.

-spec ones(term(), matrix, data_type(), orientation(), matrix_rows(), matrix_columns()) -> {ok, buffer()}.
ones(#pc_context{ref=Ctx}, matrix, float, Orientation, Rows, Cols) ->
    {ok, Buf} = new(Ctx, matrix, float, Orientation, Rows, Cols),
    ok = write(Buf, [[1.0 || _ <- lists:seq(1, Cols)] || _ <- lists:seq(1, Rows)]),
    {ok, Buf}.


-spec zeros(term(), data_type, integer()) -> {ok, buffer()}.
zeros(#pc_context{ref=Ctx}, float, Size) ->
    {ok, Buf} = new(Ctx, float),
    ok = write(Buf, [0.0 || X<-lists:seq(1, Size)]),
    {ok, Buf}.

-spec zeros(term(), matrix, data_type(), orientation(), matrix_rows(), matrix_columns()) -> {ok, buffer()}.
zeros(#pc_context{ref=Ctx}, matrix, float, Orientation, Rows, Cols) ->
    {ok, Buf} = new(Ctx, matrix, float, Orientation, Rows, Cols),
    ok = write(Buf, [[0.0 || _ <- lists:seq(1, Cols)] || _ <- lists:seq(1, Rows)]),
    {ok, Buf}.

destroy(#pc_buffer{ref=Ref}) ->
    numer_nifs:destroy_buffer(Ref),
    ok.

size(#pc_buffer{ref=Ref}) ->
    numer_nifs:buffer_size(Ref).

write(#pc_buffer{ref=Ref, data_type=Type}, Data) when Type =:= integer orelse
                                                 Type =:= string orelse
                                                 Type =:= float ->
    numer_nifs:write_buffer(Ref, Data).

read(#pc_buffer{ref=Ref}) ->
    numer_nifs:read_buffer(Ref).

duplicate(#pc_context{ref=Ctx}, #pc_buffer{ref=Ref, data_type=Type}) when Type =:= integer orelse
                                               Type =:= string orelse
                                               Type =:= float ->
    {ok, OtherBuf} = new(Ctx, Type),
    numer_nifs:copy_buffer(Ref, OtherBuf#pc_buffer.ref),
    {ok, OtherBuf}.

clear(#pc_buffer{ref=Ref}) ->
    numer_nifs:clear_buffer(Ref).


