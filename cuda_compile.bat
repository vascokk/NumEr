set CUDA_HOME="C:\Progra~1\NVIDIA~1\CUDA\v5.0"
cd c_src
REM nvcc -arch=sm_35 -O2 -c -Xcompiler  -I%CUDA_HOME%\\include numer_ops.cu -o numer_ops.o
REM nvcc -arch=sm_35 -O2 -c -Xcompiler  -I%CUDA_HOME%\\include numer_string.cu -o numer_string.o

del *.o
nvcc -O2 -c  -Xcompiler  -I%CUDA_HOME%\\include numer_kernels.cu -o numer_kernels.o
nvcc -O2 -c  -Xcompiler  -I%CUDA_HOME%\\include numer_blas.cu -o numer_blas.o
nvcc -O2 -c  -Xcompiler  -I%CUDA_HOME%\\include -I%ERL_HOME%\\usr\\include  numer_ml.cu -o numer_ml.o
nvcc -O2 -c  -Xcompiler  -I%CUDA_HOME%\\include -I%ERL_HOME%\\usr\\include  numer_cublas_wrappers.cu -o numer_cublas_wrappers.o
nvcc -O2 -c  -Xcompiler  -I%CUDA_HOME%\\include -I%ERL_HOME%\\usr\\include  numer_float_buffer.cu -o numer_float_buffer.o
nvcc -O2 -c  -Xcompiler  -I%CUDA_HOME%\\include -I%ERL_HOME%\\usr\\include  numer_matrix_float_buffer.cu -o numer_matrix_float_buffer.o

set CUDA_HOME="\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.0"