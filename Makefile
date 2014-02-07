NVFLAGS=-O3 -g -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 
CCFLAGS=-O3

# list .c and .cu source files here
CUDAFILES=gcd.cu crackRSA.cu
CPUFILES=cpu.c
LDFLAGS=-lgmp
LIBPATH=/home/clupo/gmp/lib
INCPATH=/home/clupo/gmp/include

cuda: $(CUDAFILES) 
	nvcc $(NVFLAGS) -o rsa_cuda $^ -L$(LIBPATH) -I$(INCPATH) $(LDFLAGS) -Xlinker -rpath -Xlinker $(LIBPATH)

cudaHome: $(CUDAFILES)
	nvcc $(NVFLAGS) -o rsa_cuda $^ $(LDFLAGS) 

cpu: $(CPUFILES) 
	gcc $(CCFLAGS) -std=c99 -o rsa_cpu $^ -L$(LIBPATH) -I$(INCPATH) $(LDFLAGS) -Wl,-rpath=$(LIBPATH)

cpuHome: $(CPUFILES) 
	gcc $(CCFLAGS) -o rsa_cpu $^ $(LDFLAGS)

clean: 
	rm -f *.o rsa_cuda rsa_cpu

