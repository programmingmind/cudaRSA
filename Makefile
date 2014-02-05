NVFLAGS=-O3 -g -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 
CCFLAGS=-O3
# list .c and .cu source files here
CUDAFILES=gcd.cu 
CPUFILES=classbitshifts.c

cuda: $(CUDAFILES) 
	nvcc $(NVFLAGS) -o rsa_cuda $^ 

cpu: $(CPUFILES) 
	gcc $(CCFLAGS) -o rsa_cpu $^ 

clean: 
	rm -f *.o rsa_cuda rsa_cpu

