TODO
====

Creating vectors and matrices:

``` erlang
% this is a row-major matrix:
A = [[4.0,6.0,8.0,2.0],[5.0,7.0,9.0,3.0]].

%this is a vector:
X = [2.0,5.0,1.0,7.0].

% create a CUDA context and transfer to "buffers"
{ok, Ctx} = numer_nifs:new_context().
{ok, Buf_A} = numer_nifs:new_matrix_float_buffer(Ctx, A, ?ROW_MAJOR).
{ok, Buf_X} = numer_nifs:new_float_buffer(Ctx).
numer_nifs:write_buffer(Buf_X, X).
```

There are several modules, which are wrappers for the NIF functions, like: numer\_blas.erl - for BLAS operations, numer\_buffer.erl - for operations with buffers (new, delete, read, write), etc.

Using numer\_buffer module, the above example will look like:

``` erlang
 {ok, Ctx} = numer_context:new().
 {ok, Buf_A} = numer_buffer:new(Ctx, matrix, float, row_major, A).
 {ok, Buf_X} = numer_buffer:new(Ctx, float).
 numer_buffer:write(Buf_X, X).
``` 

BLAS GEMV example:

``` erlang
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
```

See more at: http://vas.io/blog/2013/03/23/machine-learning-in-erlang-and-cuda/
