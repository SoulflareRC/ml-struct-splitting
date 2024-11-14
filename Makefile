TEST_PROGRAM="./programs/test.c" 
clean: 
	rm -rf build 

build:
	mkdir -p build
	cd build && cmake ..
	cd build && make

# Variable to control output redirection
REDIRECT_OUTPUT ?= yes

run-pass-on-test:
	@echo "Running pass on test: $(test)"
	if [ "$(REDIRECT_OUTPUT)" = "yes" ]; then \
		# cd programs && sh run.sh $(test) > run.log 2>&1 && llvm-dis *.bc; \
		cd programs && sh run.sh $(test) > run.log 2>&1; \
	else \
		# cd programs && sh run.sh $(test)  && llvm-dis *.bc; \
		cd programs && sh run.sh $(test); \
	fi

run-all: clean build run-pass-on-test




create-ir: 
	clang -emit-llvm -S $(file) -Xclang -disable-O0-optnone -o $(basename $(file)).ll

apply-pass: create-ir
	opt -load-pass-plugin=./build/structpass/StructPass.so -passes=fplicm-correctness < $(basename $(file)).ll > $(basename $(file))-opt.ll

run-pass: 
	opt -disable-output -load-pass-plugin=./build/$(pluginname)/$(pluginname).so -passes="$(passname)" $(ir)
gen-pdf: 
	opt -disable-output -passes="dot-cfg" $(ir) 
	cat .main.dot | dot -Tpdf > $(basename $(ir)).pdf
	
.PHONY: benchmark1 benchmark2 benchmark3
benchmark1:
	cd benchmark1 && sh run.sh simple
benchmark2: 
	cd benchmark2 && sh run.sh anagram 
benchmark3: 
	cd benchmark3 && sh run.sh compress

run: 
	cd benchmarks/$(BENCHMARK) && sh run.sh $(FILE) && llvm-dis *.bc
run-all-correct: 
	$(MAKE) run BENCHMARK=correctness FILE=hw2correct1
	$(MAKE) run BENCHMARK=correctness FILE=hw2correct2
	$(MAKE) run BENCHMARK=correctness FILE=hw2correct3
	$(MAKE) run BENCHMARK=correctness FILE=hw2correct4
	$(MAKE) run BENCHMARK=correctness FILE=hw2correct5
	$(MAKE) run BENCHMARK=correctness FILE=hw2correct6
# $(MAKE) gen-pdf benchmarks/$(BENCHMARK)/$(basename $(FILE)).ll 
# $(MAKE) gen-pdf benchmarks/$(BENCHMARK)/$(basename $(FILE)).fplicm.ll
	 