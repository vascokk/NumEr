

On Windows x64
--------------

``` bash
git clone git://github.com/vascokk/NumEr.git
cd NumEr

set TARGET_ARCH=x64
```

Make sure you have the following bat file:

``` bash
C:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\bin\amd64\vcvars64.bat
```

With this line inside: 

``` bash
call "C:\Program Files\Microsoft SDKs\Windows\v7.1\Bin\SetEnv.cmd" /x64
```

Run the above bat file and compile:

``` bash
rebar compile
rebar eunit suites=numer_helpers
```

TODO: MaxOS and Linux

Creating vectors and matrices
------------------------------

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
------------------

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

Using "helper" module
----------------------

Since using buffer operations can make the code awkward to read, there is also a helper module - numer\_helpers.erl, wich can be used for prototyping the algorithms. WARNING - it is extremely slow and suitable ONLY for prototyping. Here is how you can use it:

``` erlang
gemv_2_test()->
    A = [[4.0,6.0,8.0,2.0],[5.0,7.0,9.0,3.0]],
    _alpha = 1.0,
    _beta = 0.0,
    X = [2.0,5.0,1.0,7.0],
    Res = numer_helpers:gemv(no_transpose , _alpha, A, X, _beta, []),
    ?assertEqual([60.0,75.0], Res).
```

It is much more readable and useful for one-time calculations, but in the ML "training" stage (with hundreds of iterations) it will be unusable, due to the multiple buffer transfers. 

"Logistic Regression"
---------------------

There is an implementation of the Logistic Regression (without regularization) algorithm. Take a look at the numer\_logreg.erl module.

The numer\_ml.erl module contains a C++ implementation (via NIFs) of Logistic Regression, while the numer\_logreg.erl is using buffers and numer_blas NIFs. The first one I used to compare the speed between all-native and buffers+NIFs implementations.

The project is still a work in progress and needs a lot of polishing and if anyone is willing to give a hand I'll be more than happy. Any suggestions to improve the framework are also very welcome.
