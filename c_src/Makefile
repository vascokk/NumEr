CC=cl

all: compile

compile: numer_ops.o numer_blas.o numer_kernels.o numer_string.o

numer_ops.o:
	nvcc $(NVCCFLAGS) numer_ops.cu -o numer_ops.o

numer_blas.o:
	nvcc $(NVCCFLAGS) numer_blas.cu -o numer_blas.o 

numer_kernels.o:
	nvcc $(NVCCFLAGS) numer_kernels.cu -o numer_kernels.o

numer_string.o:
	nvcc $(NVCCFLAGS) numer_string.cu -o numer_string.o

clean:
	nvcc -c -clean *.cu
	rm *.o
