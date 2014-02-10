NVFLAGS=-O3 -g -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 
CCFLAGS=-O3 -std=c99

# list .c and .cu source files here
CUDAFILES=gcd.cu crackRSA.cu
CUDAOBJS=$(CUDAFILES:.cu=.o)
COMMON=common.c
CPUFILES=cpu.c
LDFLAGS=-lgmp
LIBPATH=/home/clupo/gmp/lib
INCPATH=/home/clupo/gmp/include

cuda: $(COMMON) $(CUDAOBJS)
	g++ -O3 -o rsa_cuda $^ -L$(LIBPATH) -L/usr/local/cuda/lib64 -I$(INCPATH) -lcuda -lcudart $(LDFLAGS) -Wl,-rpath=$(LIBPATH)

%.o: %.cu
	nvcc -c $(NVFLAGS) $^ -L$(LIBPATH) -I$(INCPATH) $(LDFLAGS) -Xlinker -rpath -Xlinker $(LIBPATH) 

cpu: $(CPUFILES) $(COMMON)
	gcc $(CCFLAGS) -o rsa_cpu $^ -L$(LIBPATH) -I$(INCPATH) $(LDFLAGS) -Wl,-rpath=$(LIBPATH)

cpuHome: $(CPUFILES) $(COMMON)
	gcc $(CCFLAGS) -o rsa_cpu $^ $(LDFLAGS)

clean: 
	rm -f *.o rsa_cuda rsa_cpu

