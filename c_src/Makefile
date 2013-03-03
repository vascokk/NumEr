CC=gcc

CXXFLAGS = -O3 -I/usr/local/cuda/include/  -I/usr/local/lib/erlang/usr/include/
NVCCFLAGS = $(CXXFLAGS) -c -m32 -Xcompiler -bundle,-undefined,suppress,-fno-common,-fPIC

all: compile

compile: numer_wrappers.o numer_blas.o numer_kernels.o numer_float_buffer.o numer_matrix_float_buffer.o numer_ml.o

numer_wrappers.o:
	nvcc $(NVCCFLAGS) numer_cublas_wrappers.cu -o numer_cublas_wrappers.o

numer_blas.o:
	nvcc $(NVCCFLAGS) numer_blas.cu -o numer_blas.o

numer_kernels.o:
	nvcc $(NVCCFLAGS) numer_kernels.cu -o numer_kernels.o

numer_float_buffer.o:
	nvcc $(NVCCFLAGS) numer_float_buffer.cu -o numer_float_buffer.o

numer_matrix_float_buffer.o:
	nvcc $(NVCCFLAGS) numer_matrix_float_buffer.cu -o numer_matrix_float_buffer.o

numer_ml.o:
	nvcc $(NVCCFLAGS) numer_ml.cu -o numer_ml.o



clean:
	nvcc -c -clean *.cu
	rm *.o
