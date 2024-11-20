import argparse
from pathlib import Path
import json
import analyze 
import models  
import groupers
import numpy as np 
import torch 
import torch.nn as nn 
import torch.optim as optim
from tqdm import tqdm  
import pandas as pd 
import matplotlib.pyplot as plt
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
columns = ['struct_name', 'grouping_idx', 'feature_matrix', 'grouping_vector', 'time_delta', "d1_miss_delta", "lld_miss_delta", "score"]
dataset_columns = ["feature_grouping_matrices", "target"] 
# Set pandas options to display the full DataFrame
pd.set_option('display.max_rows', None)  # No limit on the number of rows
pd.set_option('display.max_columns', None)  # No limit on the number of columns
pd.set_option('display.width', None)  # No limit on the width of the display
pd.set_option('display.max_colwidth', None)  # No limit on column width


class Runner:
    def __init__(self):
        # Parse arguments and load configuration
        self.args = self._parse_args()
        self.config = self._initialize_config(self.args)
        self._print_config()
        suppress_output = not self.config.print_pass_output 
        # self.analyzer = analyze.StructAnalyzer(source_file=self.config.source_file, build_dir=self.config.build_dir, suppress_pass_output=suppress_output) 
        self.analyzer = analyze.StructAnalyzer(config=self.config)  
        self.selector_model = models.GroupingSelector(C=groupers.GROUPERS_CNT) 
        

    def _parse_args(self):
        """ Parse command-line arguments using argparse """
        parser = argparse.ArgumentParser()
        
        # actions 
        parser.add_argument("--cleanup-files", action="store_true", help="Clean up files")
        parser.add_argument("--run-analysis-pass", action="store_true", help="Run analysis pass on C source file")
        parser.add_argument("--run-all-groupings", action="store_true", help="Run transform using all grouping methods and record time delta for all individual structs (Example: If there are C grouping methods and S structs, then there will be a total of C*S probabilities), then generate a csv recording the result")
        parser.add_argument("--run-grouping", action="store_true", help="Run analysis pass on C source file")
        parser.add_argument("--analyze-all", action="store_true", help="Generate dataset for all structs in all benchmarks")
        parser.add_argument("--transform-all", action="store_true", help="Transform all code with machine learning model prediction")
        parser.add_argument("--analyze-and-transform-all", action="store_true", help="Train and predict all")
        parser.add_argument("--predict-transform", action="store_true", help="Transform code with machine learning model prediction")
        
        parser.add_argument("--sanity-check", action="store_true", help="Check if code still works after transformation")
        parser.add_argument("--sanity-check-all", action="store_true", help="Check if code still works after transformation for all source files under folder") 
        
        parser.add_argument("--evaluate-model", action="store_true", help="Evaluate model on all benchmarks")
        
        # config parameters 
        parser.add_argument("--grouping-idx", type=int, help="selected grouping index")
        parser.add_argument("--print-pass-output", action="store_true", help="un-suppress analysis and transform pass output")
        parser.add_argument("--max-N", type=int, default=10, help="The (maximum) number of groups each struct will be divided into")
        parser.add_argument("--profile-avg-cnt", type=int, help="Number of iterations to run when profiling runtime and cache misses", default=10)  
        parser.add_argument("--train-epochs", type=int, help="Number epochs to train the model on the training data", default=100)
        parser.add_argument("--model-path", type=str, help="Path to the saved model pth file", default="model.pth")
        parser.add_argument("--source-file", type=str, help="The C source file without extension", default="programs/test_programs/test.c")
        parser.add_argument("--benchmark-dir", type=str, help="Directory for benchmark folder", default="programs/test_programs")
        parser.add_argument("--build-dir", type=str, help="The build directory used to build the pass plugin", default="build")
        
        # Parse the command-line arguments
        args = parser.parse_args()
        args = vars(args) 
        return args
    
    def _initialize_config(self, args):
        """ Initialize configuration using the parsed arguments """
        print(args) 
        config = Config(**args) 
        return config
    
    def _print_config(self):
        """ Print out the configuration details """
        print("Configuration Loaded:")
        print(self.config)

    def cleanup_files(self): 
        print("Cleaning up files")
        self.analyzer.llvm_helper.cleanup_files()

    def run_analysis_pass(self):
        print(f"Run analysis pass on source file {self.config.source_file}")
        self.analyzer.run_analysis_pass()
        print("Analysis matrices stored in field_loop_analysis.json")

    def run_all_groupings(self):
        print(f"Run all groupings on source file {self.config.source_file}")
        self.analyzer.run_all_groupings()

    def run_grouping(self, grouping_idx):
        print(f"Run grouping method on all structs of source file {self.config.source_file}")
        print("grouping idx:", grouping_idx)
        self.analyzer.run_grouping(grouping_idx)


    def get_source_files(self, directory="."): 
        folder = Path(directory) 
        files = list(folder.iterdir()) 
        files = [f for f in files if f.is_file() and f.suffix==".c"] 
        print(files) 
        return files 
    
    def analyze_all_benchmarks(self, directory=".", build_dir=Path("build")): 
        folder = Path(directory) 
        print(f"Analyzing all benchmarks under folder {folder}")
        df = pd.DataFrame(columns=columns)
        ds_df = pd.DataFrame(columns=dataset_columns) 
        files = self.get_source_files(folder) 
        for f in tqdm(files): 
            print(f) 
            if f.is_file() and f.suffix == ".c":
                print(f"Analyzing benchmark: {f.name}")
                # analyzer = analyze.StructAnalyzer(source_file=f, build_dir=build_dir) 
                config = self.config 
                config.source_file = f 
                analyzer = analyze.StructAnalyzer(config=config)  
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

    def train_selector(self, ds_df:pd.DataFrame, epochs=100):  
        # train_df, test_df = train_test_split(ds_df, test_size=0.2) 
        selector_model = models.GroupingSelector(C=groupers.GROUPERS_CNT) 
        # selector_model.train(ds_df=train_df, epochs=100) 
        # selector_model.test(ds_df=test_df)  
        # print("ds df: \n", ds_df) 
        selector_model.train(ds_df=ds_df, epochs=epochs) 
        selector_model.test(ds_df=ds_df)  
        selector_model.evaluate(ds_df=ds_df)
        selector_model.save_model(self.config.model_path)  

    def analyze_all(self):
        print("Analyzing all benchmarks and training model") 
        df, ds_df = self.analyze_all_benchmarks(directory=self.config.benchmark_dir, build_dir=self.config.build_dir)
        self.train_selector(ds_df=ds_df, epochs=self.config.train_epochs)

    def predict_and_transform_all(self, directory=".", build_dir=Path("build"), model_path=models.DEFAULT_MODEL_PATH): 
        folder = Path(directory) 
        print(f"Transforming all benchmarks under folder {folder}")
        transform_result_dict = {} 
        total_cnt = 0 
        positive_cnt = 0
        
        files = self.get_source_files(folder) 
        
        for f in tqdm(files): 
            print(f) 
            if f.is_file() and f.suffix == ".c":
                print(f"Transforming benchmark: {f.name}")
                # analyzer = analyze.StructAnalyzer(source_file=f, build_dir=build_dir) 
                config = self.config 
                config.source_file = f 
                analyzer = analyze.StructAnalyzer(config=config)  
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


    def transform_all(self):
        self.predict_and_transform_all(directory=self.config.benchmark_dir, build_dir=self.config.build_dir, model_path=self.config.model_path)

    def analyze_and_transform_all(self):
        print("Analyze and transform all")
        df, ds_df = self.analyze_all_benchmarks(directory=self.config.benchmark_dir, build_dir=self.config.build_dir)
        self.train_selector(ds_df=ds_df, epochs=self.config.train_epochs)
        self.predict_and_transform_all(directory=self.config.benchmark_dir, build_dir=self.config.build_dir, model_path=self.config.model_path)

    def predict_transform(self):
        source_file = Path(self.config.source_file)
        print(f"Predict transform for source file: {source_file}")
        self.analyzer.run_setup()
        self.analyzer.load_analysis_file()
        self.analyzer.make_prediction(self.config.model_path)

    def sanity_check(self):
        source_file = Path(self.config.source_file)
        print(f"Running sanity check for source file: {source_file}")
        self.analyzer.run_setup()
        self.analyzer.load_analysis_file()
        self.analyzer.sanity_check()

    def _sanity_check_all(self, directory=".", build_dir=Path("build")): 
        folder = Path(directory)
        print(f"Running sanity check for all source files under folder: {str(folder)}")  
        files = self.get_source_files(folder) 
        
        for f in tqdm(files): 
            print(f"Running sanity check for {str(f)}")
            analyzer = analyze.StructAnalyzer(source_file=f, build_dir=build_dir)
            analyzer.run_setup() 
            analyzer.load_analysis_file()
            analyzer.sanity_check()
    
    def sanity_check_all(self):
        print(f"Running sanity check for benchmark directory: {self.config.benchmark_dir}")
        self._sanity_check_all(self.config.benchmark_dir, self.config.build_dir)
    
    def _evaluate_model(self, ds_df:pd.DataFrame, model_path:Path=Path(models.DEFAULT_MODEL_PATH)): 
        selector_model = models.GroupingSelector(C=groupers.GROUPERS_CNT) 
        selector_model.load_model(model_path) 
        selector_model.evaluate(ds_df) 
    
    def evaluate_model(self): 
        print(f"Evaluating model on all benchmarks") 
        df, ds_df = self.analyze_all_benchmarks(directory=self.config.benchmark_dir, build_dir=self.config.build_dir)
        self._evaluate_model(ds_df, self.config.model_path) 

    def run(self):
        if self.config.cleanup_files:
            self.cleanup_files() 
        
        if self.config.run_analysis_pass:
            self.run_analysis_pass()

        if self.config.run_all_groupings:
            self.run_all_groupings()

        if self.config.run_grouping:
            self.run_grouping(self.config.grouping_idx)

        if self.config.analyze_all:
            self.analyze_all()

        if self.config.transform_all:
            self.transform_all()

        if self.config.analyze_and_transform_all:
            self.analyze_and_transform_all()            

        if self.config.predict_transform:
            self.predict_transform()

        if self.config.sanity_check:
            self.sanity_check()

        if self.config.sanity_check_all:
            self.sanity_check_all()

        if self.config.evaluate_model: 
            self.evaluate_model()


