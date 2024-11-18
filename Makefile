clean: 
	rm -rf build 

build:
	mkdir -p build
	cd build && cmake ..
	cd build && make

# Variable to control output redirection
REDIRECT_OUTPUT ?= yes

setup_venv: 
	python3 -m venv venv
	. venv/bin/activate && pip install --upgrade pip setuptools wheel
	. venv/bin/activate && pip install -r requirements.txt
	

train_predict_all: # train model and predict using programs under benchmark folder 
	python3 src/analyze.py --analyze-and-transform-all --max-N=3 --profile-avg-cnt=100 --train-epochs=1000 --benchmark-dir=programs/test_programs --build-dir=build 

predict: 
	python3 src/analyze.py --predict-transform --max-N=3 --profile-avg-cnt=10 --benchmark-dir=programs/test_programs --build-dir=build --source-file=${source_file} --model-path=${model_path} 

predict_all: 
	python3 src/analyze.py --transform-all --max-N=3 --profile-avg-cnt=10 --benchmark-dir=programs/test_programs --build-dir=build --model-path=${model_path}

sanity_check: # sanity check on whether the pipeline works on the program 
	python3 src/analyze.py --sanity-check --source-file=${source_file} 

sanity_check_all: 
	python3 src/analyze.py --sanity-check-all 

run_all_groupings: # run all groupings on the source file 
	python3 src/analyze.py --run-all-groupings --source-file=$(source_file) 

run_grouping: 
	python3 src/analyze.py --run-grouping --grouping-idx=${grouping_idx} --source-file=${source_file} 