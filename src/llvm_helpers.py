'''
This file contains LLVM helpers. 
Basically this extracts all run.sh content and converted them into python code. 
'''
import subprocess
import argparse
from pathlib import Path 
import logging 
# ACTION NEEDED: Update the paths if necessary
# BUILD_DIR = "/mnt/o/Jul28Backup/UofM-Programs/eecs583/ml-struct-splitting/build"
# PATH2LIB = BUILD_DIR+"/structpass/StructPass.so"  # Specify your build directory in the project
PATH2LIB="/structpass/StructPass.so"
ANALYSIS_PASS = "struct-analysis"
TRANSFORM_PASS = "struct-splitting"
OUTPUT_OPT = "output_opt.txt" 
OUTPUT_CORRECT = "output_correct.txt"  

TEST_PROGRAMS_BASE_DIR = "./test_programs"
import re
def subtract_dicts(lhs, rhs):
    # Initialize an empty dictionary to store the differences
    diff_dict = {}
    
    # Iterate through each key in lhs (assuming both dicts have the same keys)
    for key in lhs:
        if key in rhs:
            # Compute the difference for each key
            diff_dict[key] = lhs[key] - rhs[key]
        else:
            # If the key is missing in rhs, store the lhs value as the difference
            diff_dict[key] = lhs[key]
        # print(f"key: {key} lhs: {lhs[key]} rhs: {rhs[key]} diff: {diff_dict[key]}")
    
    return diff_dict

def parse_cachegrind_output(stdout):
    # Initialize dictionaries to store the parsed results for each cache category
    icache_stats = {
        "I_refs": 0,
        "I1_misses": 0,
        "LLi_misses": 0,
        "I1_miss_rate": 0.0,
        "LLi_miss_rate": 0.0
    }
    
    dcache_stats = {
        "D_refs": 0,
        "D1_misses": 0,
        "LLd_misses": 0,
        "D1_miss_rate": 0.0,
        "LLd_miss_rate": 0.0
    }
    
    llcache_stats = {
        "LL_refs": 0,
        "LL_misses": 0,
        "LL_miss_rate": 0.0
    }

    # Regex patterns for extracting the data
    patterns = {
        "I_refs": r"I\s+refs:\s+([\d,]+)",
        "I1_misses": r"I1\s+misses:\s+([\d,]+)",
        "LLi_misses": r"LLi\s+misses:\s+([\d,]+)",
        "I1_miss_rate": r"I1\s+miss\s+rate:\s+([\d.]+%)",
        "LLi_miss_rate": r"LLi\s+miss\s+rate:\s+([\d.]+%)",
        "D_refs": r"D\s+refs:\s+([\d,]+)",
        "D1_misses": r"D1\s+misses:\s+([\d,]+)",
        "LLd_misses": r"LLd\s+misses:\s+([\d,]+)",
        "D1_miss_rate": r"D1\s+miss\s+rate:\s+([\d.]+%)",
        "LLd_miss_rate": r"LLd\s+miss\s+rate:\s+([\d.]+%)",
        "LL_refs": r"LL\s+refs:\s+([\d,]+)",
        "LL_misses": r"LL\s+misses:\s+([\d,]+)",
        "LL_miss_rate": r"LL\s+miss\s+rate:\s+([\d.]+%)"
    }

    # Extract and process each value
    for key, pattern in patterns.items():
        match = re.search(pattern, stdout)
        if match:
            value = match.group(1).replace(",", "")
            if "%" in value:
                value = float(value.replace("%", "")) / 100  # Convert percentage to a float between 0 and 1
            else:
                value = int(value)

            # Assign the value to the correct dictionary based on the key
            if key in icache_stats:
                icache_stats[key] = value
            elif key in dcache_stats:
                dcache_stats[key] = value
            elif key in llcache_stats:
                llcache_stats[key] = value

    return icache_stats, dcache_stats, llcache_stats


def run_command(command, capture_output=False):
    logging.debug(f"Running command: {command}")
    try:
        result = subprocess.run(command, shell=True, capture_output=capture_output, text=True, check=True)
        if capture_output:
            return result.stdout, result.stderr 
        else: 
            return None 
    except subprocess.CalledProcessError as e:
        print(f"Error: Command '{e.cmd}' failed with return code {e.returncode}")
        print(f"Standard Output: {e.stdout}")
        print(f"Standard Error: {e.stderr}")
        raise  # This will re-raise the exception to stop the program
    
class LLVMHelper: 
    def __init__(self, source_file:Path, build_dir:Path):   
        source_file = Path(source_file).absolute() 
        current_dir = Path.cwd().absolute()   
        self.source_file = source_file
        self.base_dir = source_file.parent
        self.base_name = str(self.base_dir) + "/" + self.source_file.stem 
        logging.debug(f"current_dir: {current_dir} source_file: {source_file} base_dir: {self.base_dir} base name: {self.base_name}")
        self.build_dir = str(build_dir) 
    def cleanup_files(self):
        print("Cleaning up files") 
        # Delete outputs from previous runs
        files = ["default.profraw", "*_prof", "*_fplicm", "*.bc", "*.profdata", "*_output", "*.ll", "*_no_opt", "*_opt"] 
        for pattern in files:
            # Use the base_dir to form the full path for the cleanup
            file_pattern = self.base_dir / pattern
            run_command(f"rm -f {file_pattern}")
        cur_files = ["*cachegrind.out.*", "analysis.out", OUTPUT_CORRECT, OUTPUT_OPT, "default.profraw"]
        for pattern in cur_files:
            run_command(f"rm -f {pattern}")
        
    
    def emit_ll_ir(self): 
        run_command(f"llvm-dis {str(self.base_dir)}/*.bc ")
    
    def emit_bitcode(self): 
        print("Emitting bitcode") 
        # Convert source code to bitcode (IR)
        run_command(f"clang -emit-llvm -c {self.base_name}.c -Xclang -disable-O0-optnone -o {self.base_name}.bc -lm")

        # Canonicalize natural loops
        run_command(f"opt -passes='loop-simplify' {self.base_name}.bc -o {self.base_name}.ls.bc")

        # Instrument profiler passes
        run_command(f"opt -passes='pgo-instr-gen,instrprof' {self.base_name}.ls.bc -o {self.base_name}.ls.prof.bc")
    
    def generate_profile_data(self): 
        print("Generating profile data")
        # Generate binary executable with profiler embedded
        run_command(f"clang -fprofile-instr-generate {self.base_name}.ls.prof.bc -o {self.base_name}_prof -lm")

        # Run the profiler embedded executable to generate the profile data
        run_command(f"{self.base_name}_prof > {OUTPUT_CORRECT}")
        
        # Merge the profile data into a profdata file
        run_command(f"llvm-profdata merge -o {self.base_name}.profdata default.profraw")

        # Attach the profile data to the bc file using the "Profile Guided Optimization Use" pass
        run_command(f"opt -passes='pgo-instr-use' -o {self.base_name}.profdata.bc -pgo-test-profile-file={self.base_name}.profdata < {self.base_name}.ls.prof.bc > /dev/null")
    
    def generate_correct_output(self): 
        # Run the profiler embedded executable to generate the profile data
        run_command(f"{self.base_name}_no_opt > {OUTPUT_CORRECT}")    
    
    def generate_transformed_output(self): 
        # Run the optimized executable and capture output
        run_command(f"{self.base_name}_opt > {OUTPUT_OPT}")        
    
    def run_analysis_pass(self, suppress_output=False): 
        # Run analysis pass
        # if not suppress_output:
        print(f"Running Analysis Pass... Suppress output: {suppress_output}")
        
        # Redirect output to /dev/null (Unix) or nul (Windows) if suppress_output is True
        output_redirection = " > /dev/null 2>&1" if suppress_output else " > analysis.out"
        
        run_command(f"opt -load-pass-plugin='{self.build_dir+PATH2LIB}' -passes='{ANALYSIS_PASS}' {self.base_name}.profdata.bc -o {self.base_name}.opt.bc {output_redirection}")

    def run_transform_pass(self, suppress_output=False): 
        # Run transform pass
        # if not suppress_output:
        print(f"Running Transform Pass... Suppress output: {suppress_output}")
        
        # Redirect output to /dev/null (Unix) or nul (Windows) if suppress_output is True
        output_redirection = " > /dev/null 2>&1" if suppress_output else " > opt.out"
        
        run_command(f"opt -load-pass-plugin='{self.build_dir+PATH2LIB}' -passes='{TRANSFORM_PASS}' {self.base_name}.profdata.bc -o {self.base_name}.opt.bc {output_redirection}")
        
        run_command(f"llvm-dis {self.base_dir}/*.bc")

    def generate_cachegrind_report(self, executable): 
        logging.debug(f"Generating cachegrind report for executable: {executable}")
        stdout, stderr =  run_command(f"valgrind --tool=cachegrind --cachegrind-out-file=/dev/null {executable} > /dev/null", capture_output=True) 
        # print("stderr: \n", stderr)  
        cg_out = str(stderr) 
        # print(cg_out)
        cg_dict_icache, cg_dict_dcache, cg_dict_llcache =  parse_cachegrind_output(cg_out) 
        # print("cg_dict_icache: \n", json.dumps(cg_dict_icache, indent=4))
        # print("cg_dict_dcache: \n", json.dumps(cg_dict_dcache, indent=4))
        # print("cg_dict_llcache: \n", json.dumps(cg_dict_llcache, indent=4))
        return cg_dict_icache, cg_dict_dcache, cg_dict_llcache 
        
    def generate_no_opt_executable(self): 
        # # Generate binary executable before optimization
        print("Running unoptimized code...")
        executable = f"{self.base_name}_no_opt"
        run_command(f"clang {self.base_name}.ls.bc -o {executable} -lm")
        return executable
        
    def generate_opt_executable(self): 
        # # Generate binary executable after optimization
        print("Running optimized code...")
        executable = f"{self.base_name}_opt"
        run_command(f"clang {self.base_name}.opt.bc -o {executable} -lm")
        return executable 
    
    def compare_output(self): 
        no_opt_fname = OUTPUT_CORRECT 
        opt_fname = OUTPUT_OPT
        logging.debug("--------------------------------------Correct Output Content--------------------------------------"); 
        with open(no_opt_fname, "r") as f: 
            no_opt_content = f.read() 
        logging.debug(no_opt_content)
         
        logging.debug("--------------------------------------Optimized Output Content------------------------------------------------------"); 
        with open(opt_fname, "r") as f: 
            opt_content = f.read() 
        logging.debug(opt_content) 
        if opt_content == no_opt_content:
            logging.info("******************OUTPUT MATCHES!!!***********************************"); 
        else: 
            raise AssertionError("OUTPUT DOESN'T MATCH!!!!")
    
if __name__ == "__main__":         
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run LLVM passes and transformations.")
    parser.add_argument("source_file", type=str, help="The C source file without extension")
    args = parser.parse_args()

    source_file = args.source_file  # This is the `${1}` in the shell script

    llvm_helper = LLVMHelper(source_file=source_file) 
    llvm_helper.cleanup_files() 
    llvm_helper.emit_bitcode() 
    llvm_helper.generate_profile_data() 
    llvm_helper.run_analysis_pass() 
    


