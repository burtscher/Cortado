CCUDA=nvcc
CCUDAFLAGS=-O3 -DUSE_GPU -DNDEBUG -fmad=false -Xcompiler "-O3 -march=native -mno-fma -ffp-contract=off" -std=c++17
ifdef NVCC_ARCH
CCUDAFLAGS:=-arch=$(NVCC_ARCH) $(CCUDAFLAGS)
endif

store=bin/store
init_send=bin/init_send
init_recv=bin/init_recv
resi_send=bin/resi_send
resi_recv=bin/resi_recv

store_bench=bin/store_bench
init_bench=bin/init_bench
resi_bench=bin/resi_bench

.PHONY: clean

all: user benchmark

user: store init_send init_recv resi_send resi_recv

benchmark: store_bench init_bench resi_bench

clean:
	rm -f bin/*

### user mode

store: $(store)
$(store): src/cortado_store.cu
	$(CCUDA) $(CCUDAFLAGS) -I. -o $(store) src/cortado_store.cu

init_send: $(init_send)
$(init_send): src/cortado_init.cu
	$(CCUDA) -DSEND $(CCUDAFLAGS) -I. -o $(init_send) src/cortado_init.cu

init_recv: $(init_recv)
$(init_recv): src/cortado_init.cu
	$(CCUDA) -DRECV $(CCUDAFLAGS) -I. -o $(init_recv) src/cortado_init.cu

resi_send: $(resi_send)
$(resi_send): src/cortado_resi.cu
	$(CCUDA) -DSEND $(CCUDAFLAGS) -I. -o $(resi_send) src/cortado_resi.cu

resi_recv: $(resi_recv)
$(resi_recv): src/cortado_resi.cu
	$(CCUDA) -DRECV $(CCUDAFLAGS) -I. -o $(resi_recv) src/cortado_resi.cu

### benchmark mode

store_bench: $(store_bench)
$(store_bench): src/cortado_store.cu
	$(CCUDA) -DBENCHMARK $(CCUDAFLAGS) -I. -o $(store_bench) src/cortado_store.cu

init_bench: $(init_bench)
$(init_bench): src/cortado_init.cu
	$(CCUDA) -DBENCHMARK $(CCUDAFLAGS) -I. -o $(init_bench) src/cortado_init.cu

resi_bench: $(resi_bench)
$(resi_bench): src/cortado_resi.cu
	$(CCUDA) -DBENCHMARK $(CCUDAFLAGS) -I. -o $(resi_bench) src/cortado_resi.cu

