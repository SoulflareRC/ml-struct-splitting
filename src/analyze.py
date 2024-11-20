import pandas as pd 
import numpy as np 
import json 
import groupers 
import argparse 
import time 
import subprocess
from llvm_helpers import LLVMHelper
import llvm_helpers 
from pathlib import Path
from tqdm import tqdm 
import models 
from sklearn.model_selection import train_test_split 
import logging 
from config import Config 
# logging.basicConfig(level=logging.DEBUG) 
logging.basicConfig(level=logging.INFO) 
ANALYSIS_FNAME="field_loop_analysis.json"
GROUPING_FNAME="groupings.json"  
TRANSFORM_RESULT_FNAME="transform_result.json" 
ANALYSIS_RESULT_FNAME="analysis_result.json" 
STRUCT_CSV_FNAME = "_grouping_result.csv"
STRUCT_DATASET_CSV_FNAME = "_dataset_grouping_result.csv"

# TIME_EXECUTABLE_CNT=10
# DEFAULT_TIME_EXECUTABLE_CNT=10 
# MAX_N = 3 
# DEFAULT_MAX_N = 3
columns = ['struct_name', 'grouping_idx', 'feature_matrix', 'grouping_vector', 'time_delta', "d1_miss_delta", "lld_miss_delta", "score"]
dataset_columns = ["feature_grouping_matrices", "target"] 
# Set pandas options to display the full DataFrame
pd.set_option('display.max_rows', None)  # No limit on the number of rows
pd.set_option('display.max_columns', None)  # No limit on the number of columns
pd.set_option('display.width', None)  # No limit on the width of the display
pd.set_option('display.max_colwidth', None)  # No limit on column width
class StructAnalyzer: 
    # def __init__(self, source_file,build_dir:Path, max_N=DEFAULT_MAX_N, suppress_pass_output=True,  fname=ANALYSIS_FNAME): 
    def __init__(self, config:Config):
        self.config = config 
        source_file = config.source_file 
        build_dir = config.build_dir 
        max_N = config.max_N 
        suppress_pass_output = not config.print_pass_output 
        self.profile_avg_cnt = config.profile_avg_cnt 
        
        self.llvm_helper = LLVMHelper(source_file=source_file, build_dir=build_dir) 
        self.grouper_arr = groupers.get_all_groupers(max_N=max_N)
        self.columns = columns 
        source_file = Path(source_file).absolute() 
        current_dir = Path.cwd().absolute()   
        self.source_file = source_file
        self.base_dir = source_file.parent
        self.base_name = str(self.base_dir) + "/" + self.source_file.stem 
        logging.debug(f"current_dir: {current_dir} source_file: {source_file} base_dir: {self.base_dir} base name: {self.base_name}")
        
        self.suppress_pass_output = suppress_pass_output
        
    
    def sanity_check(self): 
        '''
        this is used to quickly verify if a program still works after transformation 
        this uses random grouper 
        '''    
        grouping_dict = {} 
        self.load_analysis_file(ANALYSIS_FNAME) 
        for key in self.analysis_dict.keys(): 
            matrix = self.analysis_dict[key] 
            selected_grouping_idx = 1  
            selected_grouping = self.generate_grouping(matrix, selected_grouping_idx)  
            logging.debug(f"Struct name: {key} grouping: {selected_grouping}") 
            grouping_dict[key] = selected_grouping.tolist()  
        with open(GROUPING_FNAME, "w") as f :
            json.dump(grouping_dict, f, indent=4) 
        self.run_transform() 
        self.llvm_helper.emit_ll_ir() 
        print("Sanity check passed!!!") 
         
        
    def generate_groupings(self):
        print("Generating groupings...") 
        self.load_analysis_file(ANALYSIS_FNAME) 
        self.calculate_groupings() 
        self.save_groupings_file() 
        print("Finished generating groupings!") 
    
    def run_setup(self): 
        print("Running setup")
        self.llvm_helper.cleanup_files() 
        self.llvm_helper.emit_bitcode() 
        self.llvm_helper.generate_profile_data() 
        self.llvm_helper.run_analysis_pass(suppress_output=self.suppress_pass_output) 
        
    def run_transform(self): 
        self.llvm_helper.run_transform_pass(suppress_output=self.suppress_pass_output) 
        no_opt_executable =  self.llvm_helper.generate_no_opt_executable()
        opt_executable = self.llvm_helper.generate_opt_executable() 
        self.llvm_helper.generate_correct_output() 
        self.llvm_helper.generate_transformed_output()
        self.llvm_helper.compare_output() 
        logging.debug("no opt executable: ", no_opt_executable, "opt executable: ", opt_executable) 
        return self.time_executables(noopt_fname=no_opt_executable, opt_fname=opt_executable)
        
    def load_analysis_file(self, fname=ANALYSIS_FNAME): 
        logging.debug(f"Loading analysis result from {fname}")
        with open(fname, 'r') as f: 
            analysis_dict:dict = json.load(f) 
        for key in analysis_dict.keys(): 
            field_matrix = np.array(analysis_dict[key]).T  
            analysis_dict[key] =  field_matrix 
            logging.debug(f"Struct: {key} Matrix shape: {field_matrix.shape}") 
            logging.debug(field_matrix) 
        self.analysis_dict = analysis_dict 
        
    def save_groupings_file(self, fname=GROUPING_FNAME): 
        # json_dict = { k:v.flatten().tolist() for k,v in self.grouping_dict.items() }
        grouping_dict = self.grouping_dict 
        with open(fname, "w") as f: 
            json.dump(grouping_dict, f, indent=4)  
        
        analysis_result_dict = self.analysis_result_dict 
        logging.debug("analysis result dict: \n", json.dumps(analysis_result_dict,indent=4)) 
        with open(ANALYSIS_RESULT_FNAME, "w") as f: 
            json.dump(analysis_result_dict, f, indent=4) 
                  
    def calculate_groupings(self, key=None):  
        '''This will output a grouping dict. Key is struct name and Value is grouping vector''' 
        '''If key is not none, then will only calculate the grouping for this specific key'''
        grouper_arr = self.grouper_arr 
        self.grouping_dict = {} # this is for the transformation pass 
        self.analysis_result_dict = {
            "struct_analysis" : {} 
        } # this is for later Python analysis 
        for key in self.analysis_dict.keys(): 
            matrix = self.analysis_dict[key] 
            # groupings = [ groupers.remap_to_contiguous(g.assign_groups(matrix)) for g in grouper_arr ]
            # # TODO: Selector logic, now we are just selecting random grouping 
            # selected_grouping_idx = 1  
            # selected_grouping = groupings[selected_grouping_idx]  
            
            selected_grouping_idx = 1  
            selected_grouping = self.generate_grouping(matrix, selected_grouping_idx)  
            
            logging.debug(f"Struct name: {key} grouping: {selected_grouping}") 
            self.grouping_dict[key] = selected_grouping.tolist()  
            self.analysis_result_dict["struct_analysis"][key] = {
                "struct":key, 
                "grouping_idx": selected_grouping_idx, 
                "grouping": selected_grouping.flatten().tolist(), 
                "vectors" : self.analysis_dict[key].tolist(), 
                "optimized_time_delta" : 0.0, # to be filled later 
            }
    def calculate_grouping_with_idx(self, grouping_idx:int):  
        grouping_dict = {} 
        for key in self.analysis_dict.keys(): 
            matrix = self.analysis_dict[key] 
            grouping = self.generate_grouping(matrix, grouping_idx) 
            logging.debug("grouping: ", grouping) 
            grouping_dict[key] = grouping.tolist()  
        with open(GROUPING_FNAME, "w") as f: 
            logging.debug(f"saving grouping dict to {GROUPING_FNAME}, grouping dict: \n", json.dumps(grouping_dict, indent=4))
            json.dump(grouping_dict, f, indent=4)
        return grouping_dict 
    
    def generate_grouping(self, matrix:np.array , grouper_idx:int):
        grouper = self.grouper_arr[grouper_idx]  
        return groupers.remap_to_contiguous(grouper.assign_groups(matrix)).flatten() 
    
    def time_executable(self, fname): 
        # print(f"Timing executable {fname}...")
        start_time = time.time() 
        subprocess.run([f"{fname}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        end_time = time.time() 
        elapsed_time = end_time - start_time 
        # print(f"Elapsed time for executable {fname}: {elapsed_time:.6f} seconds")
        return elapsed_time 
        
    def time_executables(self, noopt_fname, opt_fname): 
        '''
        this will return a "score" calculated from 3 data points of the executables 
        '''
        total_time_delta = 0; 
        total_d1_miss_delta = 0; # d1 cache miss delta 
        total_lld_miss_delta = 0; # last level d miss delta 
        for i in range(self.profile_avg_cnt): # run multiple times and get average 
            noopt_time = self.time_executable(noopt_fname) 
            opt_time =  self.time_executable(opt_fname) 
            time_delta = (noopt_time - opt_time) / noopt_time # delta is relative to noopt time.  
            # print(f"noopt-opt={time_delta:.6f} seconds") 
            total_time_delta = time_delta 
            
            cg_dict_icache_no_opt, cg_dict_dcache_no_opt, cg_dict_llcache_no_opt = self.llvm_helper.generate_cachegrind_report(noopt_fname) 
            cg_dict_icache_opt, cg_dict_dcache_opt, cg_dict_llcache_opt = self.llvm_helper.generate_cachegrind_report(opt_fname)
            cg_dict_icache_delta, cg_dict_dcache_delta, cg_dict_llcache_delta = llvm_helpers.subtract_dicts(cg_dict_icache_no_opt, cg_dict_icache_opt), llvm_helpers.subtract_dicts(cg_dict_dcache_no_opt, cg_dict_dcache_opt), llvm_helpers.subtract_dicts(cg_dict_llcache_no_opt, cg_dict_llcache_opt)
            
            # if cg_dict_dcache_no_opt["D1_miss_rate"] == 0: 
            #     d1_miss_delta = 0; 
            # else:   
            #     d1_miss_delta = cg_dict_dcache_delta["D1_miss_rate"] / cg_dict_dcache_no_opt["D1_miss_rate"]
            # if cg_dict_dcache_no_opt["LLd_miss_rate"] == 0: 
            #     lld_miss_delta = 0 
            # else: 
            #     lld_miss_delta = cg_dict_dcache_delta["LLd_miss_rate"] / cg_dict_dcache_no_opt["LLd_miss_rate"]  
                
            d1_miss_delta = cg_dict_dcache_delta["D1_miss_rate"] 
            lld_miss_delta = cg_dict_dcache_delta["LLd_miss_rate"] 
            total_d1_miss_delta += d1_miss_delta 
            total_lld_miss_delta += lld_miss_delta 
            
        avg_time_delta = total_time_delta / self.profile_avg_cnt 
        avg_d1_miss_delta = total_d1_miss_delta / self.profile_avg_cnt 
        avg_lld_miss_delta = total_lld_miss_delta / self.profile_avg_cnt 
        logging.debug("total D1: ", total_d1_miss_delta, " total LLD: ", total_lld_miss_delta) 
        
        # calculate score, want to maximize this  
        score = avg_time_delta + avg_d1_miss_delta + avg_lld_miss_delta 
        score = round(score, 3) 
        avg_time_delta = round(avg_time_delta, 3) 
        avg_d1_miss_delta = round(avg_d1_miss_delta, 3) 
        avg_lld_miss_delta = round(avg_lld_miss_delta, 3) 
        
        # return avg_time_delta, avg_d1_miss_delta, avg_lld_miss_delta 
        # return score
        
        return score, avg_time_delta, avg_d1_miss_delta, avg_lld_miss_delta 
    
    def search_struct(self, name) -> pd.DataFrame: 
        '''
        (we assume analysis pass has been run and we have the field_loop_analysis.json file and self.analysis_dict already)  
        
        this function will generate dataset on struct <name> with different grouping methods 
        suppose we have G grouping methods, this will generate G records
        each record contains: 
            - struct name (string)
            - feature matrix (numpy 2d array)
            - grouping vector (numpy 1d array)
            - grouping method number (int)
            - time delta  (float)
        '''
        df = pd.DataFrame(columns=self.columns) 
        
        # shared data 
        struct_name = name 
        feature_matrix = self.analysis_dict[name] 
        
        group_cnt = len(self.grouper_arr)
        
        scores = np.zeros(group_cnt) 
        feature_grouping_matrices = np.zeros((group_cnt, feature_matrix.shape[0], feature_matrix.shape[1]+1))
        
        # run each grouping method 
        for gidx in range(group_cnt): 
            print(f"Running grouping method {gidx}") 
            grouping = self.generate_grouping(feature_matrix, gidx)
            grouping_dict = {
                struct_name : grouping.tolist()
            }
            with open(GROUPING_FNAME, "w") as f: 
                json.dump(grouping_dict, f, indent=4)
            # avg_time_delta = self.run_transform() 
            score, avg_time_delta, avg_d1_miss_delta, avg_lld_miss_delta = self.run_transform() 
            # avg_time_delta = round(avg_time_delta, 3) # round to 3 digits after, more will just be deviations 
            data = {
                'struct_name': struct_name, 
                'feature_matrix': feature_matrix, 
                'grouping_vector': grouping, 
                'grouping_idx': gidx, 
                "score" : score, 
                'time_delta' : avg_time_delta, # we want the largest time delta (no_opt - opt) 
                "d1_miss_delta" : avg_d1_miss_delta, 
                "lld_miss_delta" : avg_lld_miss_delta, 
            }
            
            df = df._append(data, ignore_index=True) 

            logging.debug(f"feature matrix shape: {feature_matrix.shape} grouping shape: {grouping.shape}") 
            feature_grouping_matrix = np.hstack((grouping[..., np.newaxis], feature_matrix)) 
            logging.debug(f"feature grouping matrix shape: {feature_grouping_matrix.shape}")
            logging.debug(f"feature matrix: \n", feature_matrix)
            logging.debug(f"grouping vector: \n", grouping) 
            logging.debug(f"feature grouping matrix: \n", feature_grouping_matrix) 
            
            # time_deltas[gidx] = avg_time_delta 
            scores[gidx] = score
            feature_grouping_matrices[gidx] = feature_grouping_matrix 
                 
        # target = np.argmax(time_deltas) 
        target = np.argmax(scores) 
        logging.debug(f"scores: \n", scores);
        logging.debug(f"feature_grouping_matrices.shape: \n", feature_grouping_matrices.shape)
        logging.debug(f"target: ", target) 
        struct_result = {
            "feature_grouping_matrices" : feature_grouping_matrices, 
            "target" : target, 
        }
        # print("Struct result: ", struct_result) 
         
        # self.llvm_helper.cleanup_files() 
        return df, struct_result 
    
    def search_all_structs(self): 
        df = pd.DataFrame(columns=self.columns) 
        ds_df = pd.DataFrame(columns=dataset_columns) 
        for key in self.analysis_dict.keys(): 
            struct_df, struct_result = self.search_struct(key) 
            df = pd.concat([df, struct_df]) 
            ds_df = ds_df._append(struct_result, ignore_index=True)
            
        # print("df: \n", df) 
        # print("ds_df: \n", ds_df) 
        logging.debug(df) 
        logging.debug(ds_df) 
        df.to_csv(f"test_{STRUCT_CSV_FNAME}",index=False) 
        ds_df.to_csv(f"test_{STRUCT_DATASET_CSV_FNAME}", index=False) 
        return df, ds_df 
    
    def make_prediction(self, model_path:Path=models.DEFAULT_MODEL_PATH): 
        ''' 
        this method will transform the code using the loaded selector model 
        '''
        # load model first 
        selector_model = models.GroupingSelector(C=groupers.GROUPERS_CNT) 
        selector_model.load_model(model_path)  
        grouping_dict = {} 
        grouping_ids_dict = {} 
        # generate feature grouping matrices 
        for struct_name, feature_matrix in self.analysis_dict.items(): 
            group_cnt = len(self.grouper_arr)
            feature_grouping_matrices = np.zeros((group_cnt, feature_matrix.shape[0], feature_matrix.shape[1]+1))
            groupings = [] 
            # run each grouping method 
            for gidx in range(group_cnt): 
                logging.debug(f"Running grouping method {gidx}") 
                grouping = self.generate_grouping(feature_matrix, gidx)
                logging.debug("grouping: ", grouping, " shape: ", grouping.shape) 
                feature_grouping_matrix = np.hstack((grouping[..., np.newaxis], feature_matrix)) 
                feature_grouping_matrices[gidx] = feature_grouping_matrix 
                groupings.append(grouping) 
            selected_idx = selector_model.predict(feature_grouping_matrices=feature_grouping_matrices).item() 
            selected_grouping =  groupings[selected_idx] 
            logging.debug("struct name: ", struct_name); 
            logging.debug("Selected idx: ", selected_idx, " selected grouping: ", selected_grouping)  
            logging.debug("feature grouping matrices: \n", feature_grouping_matrices) 
            grouping_dict[struct_name] = selected_grouping.tolist()  
            grouping_ids_dict[struct_name] = selected_idx
       
        logging.debug("Final grouping ids: \n", json.dumps(grouping_ids_dict, indent=4)) 
        print("Final grouping dict: \n", json.dumps(grouping_dict, indent=4))
        print(f"Saving grouping dict to {GROUPING_FNAME}")
        with open(GROUPING_FNAME, "w") as f:
            json.dump(grouping_dict, f, indent=4) 

        # this will use the grouping dict generated before. 
        # avg_time_delta = self.run_transform() 
        score, avg_time_delta, avg_d1_miss_delta, avg_lld_miss_delta = self.run_transform() 
        
        print(f"score: {score}, avg_time_delta: {avg_time_delta},avg_d1_miss_delta: {avg_d1_miss_delta}, avg_lld_miss_delta: {avg_lld_miss_delta}") 
        return grouping_dict, grouping_ids_dict, score, avg_time_delta, avg_d1_miss_delta, avg_lld_miss_delta 
    
    def run_analysis_pass(self): 
        '''
        runs analysis pass and outputs field loop analysis json file 
        '''
        self.run_setup() 
    
    def run_all_groupings(self):
        '''
        will run all groupings on all structs of a source file 
        '''
        self.run_setup() 
        self.load_analysis_file() 
        df, ds_df =  self.search_all_structs() 
        for idx, row in df.iterrows(): 
            struct_name = row["struct_name"] 
            grouping_idx = row["grouping_idx"] 
            time_delta = row["time_delta"] 
            d1_miss_delta = row["d1_miss_delta"] 
            lld_miss_delta = row["lld_miss_delta"] 
            score = row["score"] 
            print(f"struct: {struct_name} \t grouping_idx: {grouping_idx} \t score: {score}\t time_delta: {time_delta}\t d1_miss_delta: {d1_miss_delta}\t lld_miss_delta: {lld_miss_delta}\t") 
            
    def run_grouping(self, grouping_idx:int): 
        '''
        will run the grouping_idx grouping method on all structs of the source code 
        '''
        self.run_setup() 
        self.load_analysis_file() 
        analysis_dict = {k: v.tolist() for k, v in self.analysis_dict.items()}
        grouping_dict = self.calculate_grouping_with_idx(grouping_idx) 
        score, avg_time_delta,avg_d1_miss_delta, avg_lld_miss_delta = self.run_transform() 
        print(f"analysis dict: \n", json.dumps(analysis_dict, indent=4)) 
        print(f"grouping dict: \n", json.dumps(grouping_dict, indent=4))  
        print(f"grouping: {grouping_idx} score: {score}, avg_time_delta: {avg_time_delta},avg_d1_miss_delta: {avg_d1_miss_delta}, avg_lld_miss_delta: {avg_lld_miss_delta} ") 
def get_source_files(directory="."): 
    folder = Path(directory) 
    files = list(folder.iterdir()) 
    files = [f for f in files if f.is_file() and f.suffix==".c"] 
    print(files) 
    return files 
    
def analyze_all_benchmarks(directory=".", build_dir=Path("build")): 
    folder = Path(directory) 
    print(f"Analyzing all benchmarks under folder {folder}")
    df = pd.DataFrame(columns=columns)
    ds_df = pd.DataFrame(columns=dataset_columns) 
    
    files = get_source_files(folder) 
    
    for f in tqdm(files): 
        print(f) 
        if f.is_file() and f.suffix == ".c":
            print(f"Analyzing benchmark: {f.name}")
            analyzer = StructAnalyzer(source_file=f, build_dir=build_dir) 
            analyzer.run_setup() 
            logging.debug("Setup completed!!!") 
            analyzer.load_analysis_file()
            df_sub, ds_df_sub = analyzer.search_all_structs()
            analyzer.llvm_helper.cleanup_files() # cleanup files     
            logging.debug("df_sub: \n") 
            logging.debug(df_sub) 
            df = pd.concat([df, df_sub]) 
            ds_df = pd.concat([ds_df, ds_df_sub]) 
            # print("df_sub: \n", df_sub) 
            # print("df: \n", df) 
            # print("ds_df: \n", ds_df) 
            
    df.to_csv(f"all_struct_df.csv", index=False) 
    ds_df.to_csv(f"all_struct_ds_df.csv", index=False) 
    return df, ds_df 

def train_selector(ds_df:pd.DataFrame, epochs=100): 
    # train_df, test_df = train_test_split(ds_df, test_size=0.2) 
    selector_model = models.GroupingSelector(C=groupers.GROUPERS_CNT) 
    # selector_model.train(ds_df=train_df, epochs=100) 
    # selector_model.test(ds_df=test_df)  
    # print("ds df: \n", ds_df) 
    selector_model.train(ds_df=ds_df, epochs=epochs) 
    selector_model.test(ds_df=ds_df)  
    selector_model.save_model() 

def evaluate_selector(ds_df:pd.DataFrame, model_path:Path=Path(models.DEFAULT_MODEL_PATH)): 
    selector_model = models.GroupingSelector(C=groupers.GROUPERS_CNT) 
    selector_model.load_model(model_path) 
    selector_model.evaluate(ds_df) 

def predict_and_transform_all(directory=".", build_dir=Path("build"), model_path=models.DEFAULT_MODEL_PATH): 
    folder = Path(directory) 
    print(f"Transforming all benchmarks under folder {folder}")
    transform_result_dict = {} 
    total_cnt = 0 
    positive_cnt = 0
    
    files = get_source_files(folder) 
    
    for f in tqdm(files): 
        print(f) 
        if f.is_file() and f.suffix == ".c":
            print(f"Transforming benchmark: {f.name}")
            analyzer = StructAnalyzer(source_file=f, build_dir=build_dir) 
            analyzer.run_setup() 
            print("Setup completed!!!") 
            analyzer.load_analysis_file()
            # _, grouping_ids_dict, time_delta =  analyzer.make_prediction(model_path=model_path) 
            _, grouping_ids_dict, score, avg_time_delta, avg_d1_miss_delta, avg_lld_miss_delta = analyzer.make_prediction(model_path=model_path)
            analyzer.llvm_helper.cleanup_files() # cleanup files     
            transform_result_dict[f.name] ={
                "grouping_ids_dict": grouping_ids_dict, 
                "score" : score, 
                "time_delta" : avg_time_delta, 
                "d1_miss_delta": avg_d1_miss_delta, 
                "lld_miss_delta" : avg_lld_miss_delta 
            }
            total_cnt += 1 
            if score > 0: 
                positive_cnt+=1 
    positive_rate = positive_cnt / total_cnt 
    print("final transform result: \n", json.dumps(transform_result_dict, indent=4)) 
    print(f"Total cnt: {total_cnt} Positive cnt: {positive_cnt} Positive rate: {positive_rate}")
    with open(TRANSFORM_RESULT_FNAME, "w") as f: 
        json.dump(transform_result_dict, f, indent=4) 

def sanity_check_all(directory=".", build_dir=Path("build")): 
    folder = Path(directory)
    print(f"Running sanity check for all source files under folder: {str(folder)}")  
    files = get_source_files(folder) 
    
    for f in tqdm(files): 
        print(f"Running sanity check for {str(f)}")
        analyzer = StructAnalyzer(source_file=f, build_dir=build_dir)
        analyzer.run_setup() 
        analyzer.load_analysis_file()
        analyzer.sanity_check()


# deprecated 
# def main(): 
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--run-analysis-pass", action="store_true", help="Run analysis pass on C source file") 
#     parser.add_argument("--run-all-groupings", action="store_true", help="Run transform using all grouping methods and record time delta for all individual structs (Example: If there are C grouping methods and S structs, then there will be a total of C*S probabilities), then generate a csv recording the result") 
#     parser.add_argument("--run-grouping", action="store_true", help="Run analysis pass on C source file") 
#     parser.add_argument("--grouping-idx", type=int, help="selected grouping index") 
#     parser.add_argument("--print-pass-output", action="store_true", help="un-suppress analysis and transform pass output") 
#     parser.add_argument("--max-N", type=int, default=DEFAULT_MAX_N, help="The (maximum) number of groups each struct will be divided into") 
    
#     parser.add_argument("--cleanup-files", action="store_true", help="Clean up files") 
#     parser.add_argument("--sanity-check", action="store_true", help="Check if code still works after transformation") 
#     parser.add_argument("--sanity-check-all", action="store_true", help="Check if code still works after transformation for all source files under folder") 
#     parser.add_argument("--transform-all", action="store_true", help="Transform all code with machine learning model prediction")
#     parser.add_argument("--predict-transform", action="store_true", help="Transform code with machine learning model prediction")
#     parser.add_argument("--analyze-all", action="store_true", help="Generate dataset for all structs in all benchmarks")
    
#     parser.add_argument("--analyze-and-transform-all", action="store_true", help="Train and predict all")
#     parser.add_argument("--search-all", action="store_true", help="Generate dataset for all structs") 
#     parser.add_argument("--analyze-and-transform-all", action="store_true", help="Evaluate model") 
#     parser.add_argument("--run-all", action="store_true", help="Run all") 
#     parser.add_argument("--run-setup", action="store_true", help="Generate bitcode and profile data given the C source file")  
#     parser.add_argument("--generate-groupings", action="store_true", help="Generate groupings.json for transformation pass") 
#     parser.add_argument("--time-executables", action="store_true", help="Generate time data for unoptimized and optimized executables") 
#     parser.add_argument("--opt-fname", type=str, help="Optimized executable") 
#     parser.add_argument("--noopt-fname", type=str, help="Unoptimized executable") 
#     parser.add_argument("--source-file", type=str, help="The C source file without extension", default="programs/test_programs/test.c") 
#     parser.add_argument("--benchmark-dir", type=str, help="Directory for benchmark folder", default="programs/test_programs")  
#     parser.add_argument("--profile-avg-cnt", type=int, help="Number of iterations to run when profiling runtime and cache misses", default=DEFAULT_TIME_EXECUTABLE_CNT)  
#     parser.add_argument("--train-epochs", type=int, help="Number epochs to train the model on the training data", default=100)  
#     parser.add_argument("--model-path", type=str, help="Path to the saved model pth file", default=models.DEFAULT_MODEL_PATH)  
#     parser.add_argument("--build-dir", type=str, help="The build directory used to build the pass plugin", default="build") 
#     args = parser.parse_args() 
     
#     #  make use of args 
#     source_file = args.source_file 
#     source_file = Path(source_file) 
#     print_pass_output = args.print_pass_output 
#     suppress_output = not print_pass_output 
#     benchmark_dir = Path(args.benchmark_dir)
#     build_dir = Path(args.build_dir)  
#     profile_avg_cnt = int(args.profile_avg_cnt)  
#     TIME_EXECUTABLE_CNT = profile_avg_cnt 
#     train_epochs = args.train_epochs 
#     model_path = Path(args.model_path) 
#     print("Print pass output: ", print_pass_output)
#     print("Benchmark directory: ", benchmark_dir)
#     max_N = int(args.max_N)  
#     print("max_N: ", max_N)
#     start = time.time() 
#     if args.cleanup_files: 
#         print("Cleaning up files") 
#         analyzer = StructAnalyzer(source_file=source_file, build_dir=build_dir, suppress_pass_output=suppress_output) 
#         analyzer.llvm_helper.cleanup_files() 
#     if args.run_analysis_pass: 
#         print(f"Run analysis pass on source file {source_file}")
#         analyzer = StructAnalyzer(source_file=source_file, build_dir=build_dir, suppress_pass_output=suppress_output) 
#         analyzer.run_analysis_pass() 
#         print(f"Analysis matrices stored in field_loop_analysis.json") 
#     if args.run_all_groupings: 
#         print(f"Run all groupings on source file {source_file}") 
#         analyzer = StructAnalyzer(source_file=source_file, build_dir=build_dir, suppress_pass_output=suppress_output)
#         analyzer.run_all_groupings() 
#     if args.run_grouping: 
#         print(f"Run grouping method on all structs of source file {source_file}") 
#         analyzer = StructAnalyzer(source_file=source_file, build_dir=build_dir, suppress_pass_output=suppress_output)
#         print("grouping idx: ", args.grouping_idx)
#         grouping_idx = args.grouping_idx   
#         analyzer.run_grouping(grouping_idx=grouping_idx)
#     if args.run_setup: 
#         print("Run Setup") 
#         analyzer = StructAnalyzer(source_file=source_file, build_dir=build_dir,suppress_pass_output=suppress_output)
#         analyzer.run_setup() 
#     if args.generate_groupings:
#         print("Generate groupings") 
#         analyzer = StructAnalyzer(source_file=source_file, build_dir=build_dir,suppress_pass_output=suppress_output)
#         analyzer.generate_groupings() 
#     if args.time_executables: 
#         print("Time executables")
#         analyzer = StructAnalyzer(source_file=source_file, build_dir=build_dir,suppress_pass_output=suppress_output)
#         analyzer.time_executables(args.noopt_fname, args.opt_fname) 
#     if args.run_all: 
#         print("Run all") 
#         analyzer = StructAnalyzer(source_file=source_file, build_dir=build_dir, suppress_pass_output=suppress_output)
#         analyzer.run_setup() 
#         analyzer.generate_groupings() 
#         analyzer.run_transform() 
#     if args.search_all: 
#         print("Search all") 
#         analyzer = StructAnalyzer(source_file=source_file, build_dir=build_dir, suppress_pass_output=suppress_output)
#         analyzer.run_setup() 
#         analyzer.load_analysis_file() 
#         analyzer.search_all_structs()
#     if args.analyze_all:
#         print("Analyze all" )
#         df, ds_df = analyze_all_benchmarks(directory=benchmark_dir, build_dir=build_dir)
#         train_selector(ds_df=ds_df, epochs=train_epochs)   
#     if args.transform_all: 
#         predict_and_transform_all(directory=benchmark_dir, build_dir=build_dir, model_path=model_path)  
#     if args.analyze_and_transform_all: 
#         print("Analyze and transform all" )
#         df, ds_df = analyze_all_benchmarks(directory=benchmark_dir, build_dir=build_dir)
#         train_selector(ds_df=ds_df, epochs=train_epochs)
#         predict_and_transform_all(directory=benchmark_dir, build_dir=build_dir)
#     if args.predict_transform:
#         source_file = Path(source_file) 
#         print(f"Predict transform for source file: {source_file}") 
#         analyzer = StructAnalyzer(source_file=source_file, build_dir=build_dir, suppress_pass_output=suppress_output)
#         analyzer.run_setup() 
#         analyzer.load_analysis_file()
#         analyzer.make_prediction(model_path) 
#     if args.sanity_check: 
#         source_file = Path(source_file) 
#         print(f"Running sanity check for source file: {source_file}")
#         analyzer = StructAnalyzer(source_file=source_file, build_dir=build_dir, suppress_pass_output=suppress_output)
#         analyzer.run_setup() 
#         analyzer.load_analysis_file()
#         analyzer.sanity_check()
#     if args.sanity_check_all: 
#         print(f"Running sanity check for benchmark directory: {benchmark_dir}") 
#         sanity_check_all(benchmark_dir, build_dir) 
        
        
#     duration = time.time() - start 
#     print(f"Run took {duration} seconds")  
# if __name__ == "__main__": 
#     main() 

