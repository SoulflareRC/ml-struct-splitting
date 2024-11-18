# Struct-Splitting with Machine Learning 
> This project is submitted to EECS583 FA24's final project.
> 
This project features: 
1. An LLVM pass implementing struct splitting with support of static and dynamic struct arrays.
2. A machine learning system that selects the best way of splitting each struct in a source file and performs the LLVM pass to transform the LLVM IR.     
![Revised Project Proposal](https://github.com/user-attachments/assets/09c8549c-a204-481d-b077-9dc46cc47079)

## How this works 
1. Generate LLVM profile data from the source code
2. Apply Analysis pass to generate a field-loop access matrix JSON file for each struct in the source code
3. Run the ML model in Python to predict the grouping for each struct, and output this into a JSON file
4. Apply Transform pass to read the grouping JSON file produced by the Python program and transform the source code

## Usage
The main Python scripts used in this project are under the `src` folder. The LLVM passes were written under the `structpass` folder. The benchmarks used in this project are under the `programs` folder. 

Although the system is driven by `src/analyze.py`,  a `Makefile` is included to make usage of this project easier. 
- `make build`: Build the LLVM pass plugin
- `make clean`: Clean the `build` directory (for rebuilding the LLVM pass plugins)
- `make setup_venv`: Set up the Python virtual environment and install dependencies
- `make train_predict_all`: This runs the entire train + predict process using the benchmark programs.
- `make predict_all model_path=<model_path>`: This runs prediction on all benchmark programs using the specified model.
- `make predict source_file=<source_file> model_path=<model_path>`: This runs prediction on the source file using the specified model.
- `make sanity_check source_file=<source_file>`: This runs a basic sanity check on the source file to see if the system runs correctly on the source file.
- `make sanity_check_all`: This runs sanity check on all source files under `programs/test_programs` to see if the system works correctly on all source files under the folder.
- `make run_grouping grouping_idx=<grouping_idx> source_file=<source_file>`: This runs the specified grouping method on the specified source file, and outputs score, time delta, d1 miss rate, lld miss rate, and the grouping vectors. 
- `make run_all_groupings source_file=<source_file>`: This runs all grouping methods on each struct defined in the source file and outputs the scores of each grouping method.


## Score 
The "best" grouping method is determined by a score obtained by running the transformed executable using each grouping method. The score consists of 3 parts: 
- `time_delta`: This is the relative execution time difference between unoptimized code vs optimized code using the selected grouping method. The formula used is `(no_opt_exec_time-opt_exec_time)/no_opt_exec_time`
- `d1_miss_rate`: This comes from cachegrind output. This is the L1 data cache miss rate difference between unoptimized code vs optimized code using the selected grouping method. The formula used is `no_opt_miss_rate/opt_miss_rate`.
- `lld_miss_rate`: This comes from cachegrind output. This is the last level data cache miss rate difference between unoptimized code vs optimized code using the selected grouping method. The foumula is the same as `d1_miss_rate`.
All above metrics are taken from the average of running the executables `profile-avg-cnt` times, and the final score is simply `time_delta+d1_miss_rate+lld_miss_rate`. This is used as a criterion of selecting the best grouping method.

## Grouping Methods 
A total of 7 grouping methods were implemented in `src/groupers.py`: 
1. `NoSplittingGrouper`: This assigns all fields in the struct to group 0.
2. `RandomGrouper`: This randomly assigns fields in the struct to `max_N` groups.
3. `HotnessGrouper`: This aggregates fields by their access pattern bit vectors and group fields by average aggregated hotness in descending order.
4. `KMeansGrouper`: This groups fields with KMeans clustering using their feature vector from field loop access matrix.
5. `AgglomerativeGrouper`: This groups fields with KMeans clustering using their feature vector from field loop access matrix.
6. `SpectralGrouper`: This groups fields with Spectral clustering using their feature vector from field loop access matrix.
7. `GMMGrouper`: This groups fields with GMM clustering using their feature vector from field loop access matrix.
