import argparse
from pathlib import Path
import json
import analyze 
import models  
import groupers
import numpy as np 
from tqdm import tqdm  
import pandas as pd 
import matplotlib.pyplot as plt
import logging 
from config import Config 
from datetime import datetime
import matplotlib.cm as cm
import time 
from sklearn.model_selection import train_test_split 
from collections import Counter 
# logging.basicConfig(level=logging.DEBUG) 
logging.basicConfig(level=logging.INFO) 
ANALYSIS_FNAME="field_loop_analysis.json"
GROUPING_FNAME="groupings.json"  
TRANSFORM_RESULT_FNAME="transform_result.json" 
ANALYSIS_RESULT_FNAME="analysis_result.json" 
STRUCT_CSV_FNAME = "_grouping_result.csv"
STRUCT_DATASET_CSV_FNAME = "_dataset_grouping_result.csv"
TRANSFORM_RESULT_CSV_FNAME="_transform_result.csv" 
columns = ['struct_name', 'grouping_idx', 'feature_matrix', 'grouping_vector', 'time_delta', "d1_miss_delta", "lld_miss_delta", "score"]
dataset_columns = ["feature_grouping_matrices", "target"] 
# Set pandas options to display the full DataFrame
# pd.set_option('display.max_rows', None)  # No limit on the number of rows
# pd.set_option('display.max_columns', None)  # No limit on the number of columns
# pd.set_option('display.width', None)  # No limit on the width of the display
# pd.set_option('display.max_colwidth', None)  # No limit on column width

class Runner:
    def __init__(self, config_override:dict={}): 
        # Parse arguments and load configuration
        print("Init") 
        self.args = self._parse_args()
        print("Args parsed") 
        self.config = self._initialize_config(self.args)
        print("Config initialized") 
        if config_override != {}: 
            self.config.update_from_dict(config_override) 
            print(f"Config overriden with values: \n{json.dumps(config_override, indent=4)}") 
         
        
        
        self._print_config()  
        self.config.save_to_file(self.config.experiment_dir.joinpath(f"config.json"))
        print("Config saved")  
        # self.analyzer = analyze.StructAnalyzer(source_file=self.config.source_file, build_dir=self.config.build_dir, suppress_pass_output=suppress_output) 
        self.analyzer = analyze.StructAnalyzer(config=self.config)  
        self.selector_model = models.GroupingSelector(C=groupers.GROUPERS_CNT, L=self.config.loop_cnt, hidden_size=self.config.hidden_size, rnn_type=self.config.rnn_type)   

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
        parser.add_argument("--analyze-transform-result", action="store_true", help="Takes a given transform result csv file and analyzes it and generates charts") 
        parser.add_argument("--analyze-dataset", action="store_true", help="Analyze the ds_df CSV file to visualize the dataset") 
        
        
        # config parameters 
        parser.add_argument("--config-path", type=str, default=None, help="If specified, will load config from this json file, all other parameters EXCEPT experiment name will be overwritten")  
        parser.add_argument("--experiment-name", type=str, default=None, help="If specified, will use the this as the name of the experiment.") 
        parser.add_argument("--analysis-df-path", type=str, default=None, help="The path to the CSV file containing the result of struct analysis. If this is specified then the training/prediction process will skip the analysis and just read from this file.")
        parser.add_argument("--analysis-ds-df-path", type=str, default=None, help="The path to the ds_df CSV file") 
        parser.add_argument("--rnn-type", type=str, default="GRU", help="The type of RNN used by the model") 
        parser.add_argument("--hidden-size", type=int, default=64, help="The hidden size of the model") 
        parser.add_argument("--transform-result-path", type=str, help="The transform result csv file", default="test_"+TRANSFORM_RESULT_CSV_FNAME)
        parser.add_argument("--grouping-idx", type=int, help="selected grouping index")
        parser.add_argument("--print-pass-output", action="store_true", help="un-suppress analysis and transform pass output")
        parser.add_argument("--max-N", type=int, default=3, help="The (maximum) number of groups each struct will be divided into")
        
        parser.add_argument("--sanity-check-N", type=int, default=5, help="The amount of times sanity check will be ran on the benchmark")
        parser.add_argument("--profile-avg-cnt", type=int, help="Number of iterations to run when profiling runtime and cache misses", default=10)  
        parser.add_argument("--train-epochs", type=int, help="Number epochs to train the model on the training data", default=1000)
        parser.add_argument("--model-path", type=str, help="Path to the saved model pth file", default=None)
        parser.add_argument("--source-file", type=str, help="The C source file without extension", default="programs/test_programs/test.c")
        parser.add_argument("--benchmark-dir", type=str, help="Directory for benchmark folder", default="programs/test_programs")
        parser.add_argument("--build-dir", type=str, help="The build directory used to build the pass plugin", default="build")
        
        parser.add_argument("--loop-cnt", type=int, help="L hottest loops", default=10) 
        parser.add_argument("--feature-mode", type=str, help="LOOP or BB", default="LOOP") 
        
        # Parse the command-line arguments
        args = parser.parse_args()
        args = vars(args) 
        return args
    
    def _initialize_config(self, args):
        """ Initialize configuration using the parsed arguments """
        print(args) 
        config = Config(**args) 
        self.config = config 
        setattr(config, "experiment_dir", self.get_experiment_dir()) 
        setattr(config, "transform_result_path", config.experiment_dir.joinpath(f"test_{TRANSFORM_RESULT_CSV_FNAME}")) 
        setattr(config, "model_path", config.experiment_dir.joinpath(f"model.pth")) 
        setattr(config, "analysis_ds_df_path", config.experiment_dir.joinpath(f"{Path(config.benchmark_dir).name}_df.csv"))
        setattr(config, "analysis_df_path", config.experiment_dir.joinpath(f"{Path(config.benchmark_dir).name}_ds_df.csv"))
        print(f"config model path: {config.model_path}") 
        if config.config_path != None: 
            override_keys = [
                "rnn_type", "hidden_size", 
                "transform_result_path", "model_path", "analysis_ds_df_path", "analysis_df_path",
                "benchmark_dir", "build_dir", 
                "max_N", "profile_avg_cnt", "train_epochs", "loop_cnt", "feature_mode"  
            ]
            with open(config.config_path, "r") as f: 
                config_dict = json.load(f) 
            config.update_from_dict(config_dict, override_keys=override_keys)   
            print(f"Overriden config with {config.config_path}")
            
        self.config = config 
        return config
    
    def _print_config(self):
        """ Print out the configuration details """
        print("Configuration Loaded:")
        print(self.config)
    
    def get_experiment_dir(self, unnamed=False) -> Path: 
        '''
        This function will create a folder under experiment base path (with parents=True) and return the Path to it. 
        If unnamed is True, then the folder's name is the current timestamp using YYYY-MM-DD-HH-MM-SS. 
        Otherwise, this will compose the folder's name using self.config's attributes including: 
        - hidden_size (int) 
        - rnn_type (str) 
        - max_N (int)
        - profile_avg_cnt (int) 
        - train_epochs (int) 
        '''
        # Base directory for experiments
        experiment_base_path = Path("experiments")
        
        # Determine folder name
        if unnamed:
            # Use current timestamp for folder name
            folder_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        else:
            if self.config.experiment_name is not None: 
                folder_name = self.config.experiment_name
            else: 
                # Compose folder name using config attributes
                folder_name = (
                    f"hidden_size-{self.config.hidden_size}_"
                    f"rnn_type-{self.config.rnn_type}_"
                    f"max_N-{self.config.max_N}_"
                    f"profile_avg_cnt-{self.config.profile_avg_cnt}_"
                    f"train_epochs-{self.config.train_epochs}_"
                    f"loop_cnt-{self.config.loop_cnt}_" 
                    f"feature_mode-{self.config.feature_mode}" 
                )
        
        # Create the full path
        experiment_dir = experiment_base_path / folder_name
        
        # Create the directory with parents=True
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        return experiment_dir
        

    def cleanup_files(self): 
        print("Cleaning up files")
        self.analyzer.llvm_helper.cleanup_files()

    def run_analysis_pass(self):
        print(f"Run analysis pass on source file {self.config.source_file}")
        self.analyzer.run_analysis_pass()
        self.analyzer.llvm_helper.emit_ll_ir() 
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
        return df, ds_df 

    def train_selector(self, ds_df:pd.DataFrame, epochs=100):  
        train_df, test_df = train_test_split(ds_df, test_size=0.2) 
        selector_model = models.GroupingSelector(C=groupers.GROUPERS_CNT, L=self.config.loop_cnt, hidden_size=self.config.hidden_size)  
        print(f"Dataset size: {len(ds_df)} Training Size: {len(train_df)} Testing Size: {len(test_df)}") 
        # selector_model.train(ds_df=ds_df, epochs=epochs) 
        # selector_model.test(ds_df=ds_df)  
        # selector_model.evaluate(ds_df=ds_df, output_dir=self.config.experiment_dir) 
        selector_model.train(ds_df=train_df, epochs=epochs) 
        selector_model.test(ds_df=test_df)  
        selector_model.evaluate(ds_df=test_df, output_dir=self.config.experiment_dir) 
        # selector_model.save_model( self.config.experiment_dir.joinpath(self.config.model_path)) 
        selector_model.save_model(self.config.model_path)  

    def get_all_benchmark_ds_df(self): 
        if self.config.analysis_df_path is not None: 
            print(f"Reading all benchmark analysis from {self.config.analysis_df_path}") 
            ds_df = pd.read_csv(self.config.analysis_df_path, dtype={"feature_grouping_matrices":str, "target": int}) 
            ds_df['feature_grouping_matrices'] = ds_df['feature_grouping_matrices'].apply(
                lambda x : np.array(json.loads(str(x))) 
            )
        else: 
            df, ds_df = self.analyze_all_benchmarks(directory=self.config.benchmark_dir, build_dir=self.config.build_dir)
            df.to_csv(self.config.analysis_ds_df_path) 
            self.analyze_dataset(df) 
        print("Column Types:")
        print(ds_df.dtypes)
        
        processed_ds_df = ds_df.copy() # make a copy and save this to csv  
        processed_ds_df["feature_grouping_matrices"] = processed_ds_df["feature_grouping_matrices"].apply(
            lambda x: json.dumps(x.tolist()) 
        ) 
        processed_ds_df.to_csv( self.config.analysis_df_path, index=False) 
        print(processed_ds_df.head()) 
        
        return ds_df 
        
    def analyze_all(self):
        print("Analyzing all benchmarks and training model") 
        ds_df = self.get_all_benchmark_ds_df() 
        self.train_selector(ds_df=ds_df, epochs=self.config.train_epochs) 
        
    def analyze_and_generate_bar_charts(self, transform_result_df: pd.DataFrame, output_dir: Path = Path(".")):
        """
        Analyzes the statistics of scores, time_delta, d1_miss_delta, lld_miss_delta,
        and generates bar charts for each of these columns, saving them to files.
        
        Parameters:
        - transform_result_df: A DataFrame containing the columns 'file', 'score', 'time_delta', 
                                'd1_miss_delta', and 'lld_miss_delta'.
        - output_dir: Directory path where the bar charts should be saved.
        """
        
        # Ensure the output directory exists
        if not output_dir.exists(): 
            output_dir.mkdir(parents=True, exist_ok=True) 
        
        # Calculate basic statistics
        stats = transform_result_df[['score', 'time_delta', 'd1_miss_delta', 'lld_miss_delta']].describe().round(4) 
        # Calculate sign counts
        sign_counts = transform_result_df[['score', 'time_delta', 'd1_miss_delta', 'lld_miss_delta']].apply(
            lambda col: pd.Series({
                "positive_count": (col > 0).sum(),
                "zero_count": (col == 0).sum(),
                "negative_count": (col < 0).sum(),
                "positive_percentage": round((col > 0).mean() * 100, 4),  # Round to 4 decimals
                "zero_percentage": round((col == 0).mean() * 100, 4),      # Round to 4 decimals
                "negative_percentage": round((col < 0).mean() * 100, 4),   # Round to 4 decimals
            })
        )


        # Transpose `sign_counts` so it matches the structure of `stats`
        # sign_counts = sign_counts.T
        print(sign_counts) 
        summary = pd.concat([stats, sign_counts]) 
        summary.to_csv(self.config.experiment_dir.joinpath(f"transform_summary.csv")) 
        
        grouper_names = groupers.get_all_grouper_names() 
        
        
        ### Task 1: Overall Counts and Pie Chart
        # Flatten all dictionaries into a single list of group numbers
        all_groups = [group for row in transform_result_df["grouping_ids_dict"] for group in row.values()]
        print(all_groups)
        group_counts = Counter(all_groups)

        # Convert counts to DataFrame for further analysis
        group_stats_df = pd.DataFrame.from_dict(group_counts, orient='index', columns=['count'])
        group_stats_df['percentage'] = (group_stats_df['count'] / group_stats_df['count'].sum()) * 100
        group_stats_df = group_stats_df.sort_index()
        print(group_stats_df)

        # Generate consistent color mapping for groups
        unique_group_indices = sorted(group_stats_df.index)
        color_mapping = {group: plt.cm.tab20(i % 20) for i, group in enumerate(unique_group_indices)}

        # Create labels with group index and name
        group_labels = [f"{index} ({grouper_names[index]})" for index in group_stats_df.index]

        # Plot Pie Chart with consistent color mapping
        plt.figure(figsize=(8, 8))
        plt.pie(group_stats_df['count'], labels=group_labels, autopct='%1.1f%%', startangle=140, colors=[color_mapping[group] for group in group_stats_df.index])
        plt.title("Group Distribution Across All Records")
        plt.tight_layout()
        plt.savefig(output_dir.joinpath("grouping_percentages.png"))
        plt.close()


        ### Task 2: Per-row Group Counts and Stacked Bar Chart

        # Calculate group counts for each row
        row_group_counts = transform_result_df["grouping_ids_dict"].apply(lambda x: Counter(x.values()))

        # Normalize the row counts into a DataFrame
        row_group_df = pd.DataFrame(list(row_group_counts)).fillna(0).astype(int)

        # Plot Stacked Bar Chart with consistent color mapping
        ax = row_group_df.plot(kind='bar', stacked=True, figsize=(10, 6), color=[color_mapping[group] for group in row_group_df.columns])

        # Set title and labels
        plt.title("Group Number Composition for Each Benchmark")
        plt.xlabel("File")
        plt.ylabel("Count of Group Numbers")

        # Set x-axis to use "file" field for labeling
        plt.xticks(ticks=range(len(transform_result_df)), labels=transform_result_df["file"], rotation=90)

        # Customize legend: sorted by group name, using the same color mapping
        handles, labels = ax.get_legend_handles_labels()
        sorted_labels = [f"Group {label}" for label in sorted(labels, key=int)]  # Sort the group names by their index
        ax.legend(handles, sorted_labels, title="Group Numbers")

        # Layout adjustments and saving the plot
        plt.tight_layout()
        plt.savefig(output_dir.joinpath("grouping_distribution.png"))
        plt.close()


        # Print out the statistics (optional)
        print("Statistics Summary:\n", summary)  
        
        # Create bar charts for each column
        # Loop through each row in the dataframe
        # List of columns to generate bar charts for
        # List of columns to generate bar charts for
        columns = ['score', 'time_delta', 'd1_miss_delta', 'lld_miss_delta']
        
        # Loop through each of the columns (e.g., score, time_delta, etc.)
        for column in columns:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Extract the data for the current column and file names
            bars = ax.bar(transform_result_df['file'], transform_result_df[column], color='skyblue')
            
            # Annotate each bar with its value
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height, 
                        f'{height:.2f}', ha='center', va='bottom', fontsize=10, color='black')
            
            # Set chart title and labels
            ax.set_title(f'{column} for Each File', fontsize=16)
            ax.set_ylabel(column, fontsize=12)
            ax.set_xlabel('File', fontsize=12)
            ax.set_xticklabels(transform_result_df['file'], rotation=45, ha='right')
            
            # Save the figure to the output directory
            chart_fname = output_dir.joinpath(f'{column}_bar_chart.png')
            plt.tight_layout()
            plt.savefig(chart_fname)
            plt.close()
        
        print(f"Bar charts saved to: {output_dir}")
    
    
    def analyze_transform_result(self): 
        print("Analyzing transform result") 
        df = pd.read_csv(self.config.transform_result_path)  
        # print(df.head()) 
        df["grouping_ids_dict"] = df["grouping_ids_dict"].apply(lambda x: json.loads(x)) 
        self.analyze_and_generate_bar_charts(df, output_dir=self.config.experiment_dir)      

    def _analyze_dataset(self, df:pd.DataFrame): 
        self._visualize_struct_grouping(df) 
        self._visualize_struct_best_grouping(df) 
        self._visualize_grouping_percentage(df) 
        self._visualize_struct_posneg_percentage(df) 
        self._visualize_grouping_idx_percentage(df) 
        
    def _visualize_grouping_idx_percentage(self, df):
        """
        Visualize the percentage of positive vs. negative scores for each grouping_idx
        as horizontal bar charts using an orange and blue color scheme.

        :param df: A Pandas DataFrame with columns:
                'struct_name', 'grouping_idx', 'score'
        """
        # Calculate positive and negative percentages by grouping_idx
        grouping_data = df.groupby('grouping_idx').apply(
            lambda x: {
                'positive': (x['score'] > 0).mean() * 100,  # Percentage positive
                'negative': (x['score'] <= 0).mean() * 100  # Percentage negative
            }
        ).reset_index()

        # Convert to a proper DataFrame for plotting
        grouping_data = pd.DataFrame(grouping_data.to_dict('records'))
        grouping_data.columns = ['grouping_idx', 'percentages']
        grouping_data['positive'] = grouping_data['percentages'].apply(lambda x: x['positive'])
        grouping_data['negative'] = grouping_data['percentages'].apply(lambda x: x['negative'])

        # Plot horizontal bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        y_positions = np.arange(len(grouping_data))  # One bar per grouping_idx
        bar_height = 0.4  # Height of bars

        # Positive and negative bars
        ax.barh(y_positions, grouping_data['positive'], bar_height, label='Positive %', color='blue')
        ax.barh(y_positions, -grouping_data['negative'], bar_height, label='Negative %', color='orange')

        # Label and style the chart
        ax.set_yticks(y_positions)
        ax.set_yticklabels([f"Grouping Idx {idx}" for idx in grouping_data['grouping_idx']])
        ax.set_xlabel('Percentage (%)')
        ax.set_ylabel('Grouping Index')
        ax.set_title('Percentage of Positive vs. Negative Scores by Grouping Index')
        ax.axvline(0, color='black', linewidth=0.8, linestyle='--')  # Divider for positive/negative
        ax.legend(loc='best')
        plt.tight_layout()

        # Save the figure
        plt.savefig(self.config.experiment_dir.joinpath("dataset_grouping_posneg.png"))
    
    
    def _visualize_struct_posneg_percentage(self, df):
        """
        Visualize the percentage of grouping_idx results (positive vs. negative scores) for each struct
        as horizontal bar charts with updated color scheme.

        :param df: A Pandas DataFrame with columns:
                'struct_name', 'grouping_idx', 'score'
        """
        # Prepare data for positive and negative percentages
        struct_grouped = df.groupby('struct_name').apply(
            lambda x: {
                'positive': (x['score'] > 0).mean() * 100,  # Percentage positive
                'negative': (x['score'] <= 0).mean() * 100  # Percentage negative
            }
        ).reset_index()

        # Split into separate columns
        struct_grouped = pd.DataFrame(struct_grouped.to_dict('records'))
        struct_grouped.columns = ['struct_name', 'percentages']
        struct_grouped['positive'] = struct_grouped['percentages'].apply(lambda x: x['positive'])
        struct_grouped['negative'] = struct_grouped['percentages'].apply(lambda x: x['negative'])

        # Plot horizontal bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        y_positions = np.arange(len(struct_grouped))  # One bar per struct name
        bar_height = 0.4  # Height of bars

        # Positive and negative bars with updated colors
        ax.barh(y_positions, struct_grouped['positive'], bar_height, label='Positive %', color='blue')
        ax.barh(y_positions, -struct_grouped['negative'], bar_height, label='Negative %', color='orange')

        # Label and style the chart
        ax.set_yticks(y_positions)
        ax.set_yticklabels(struct_grouped['struct_name'])
        ax.set_xlabel('Percentage (%)')
        ax.set_ylabel('Struct Name')
        ax.set_title('Percentage of Positive vs. Negative Scores by Struct Name')
        ax.axvline(0, color='black', linewidth=0.8, linestyle='--')  # Divider for positive/negative
        ax.legend(loc='best')
        plt.tight_layout()

        # Save the figure
        plt.savefig(self.config.experiment_dir.joinpath("dataset_struct_posneg.png"))
    
    def _visualize_struct_grouping(self, df:pd.DataFrame):  
            """
            Visualize scores for each struct as horizontal bar charts.
            Each struct displays multiple bars for different grouping_idx and their scores.
            
            :param df: A Pandas DataFrame with columns: 
                    'struct_name', 'grouping_idx', 'score'
            """
            # Sort the dataframe by struct_name and grouping_idx for consistent plotting
            df = df.sort_values(by=['struct_name', 'grouping_idx'])

            # Get unique structs and indices
            structs = df['struct_name'].unique()
            grouping_indices = df['grouping_idx'].unique()

            # Initialize a figure
            fig, ax = plt.subplots(figsize=(10, 6))

            # Set up y-axis positions
            y_positions = np.arange(len(structs))  # Positions for each struct
            bar_height = 0.15  # Height of each bar
            offsets = np.arange(-(len(grouping_indices) // 2), len(grouping_indices) // 2 + 1) * bar_height

            # Plot the bars
            for i, g_idx in enumerate(grouping_indices):
                # Extract rows for the current grouping index
                current_group = df[df['grouping_idx'] == g_idx]

                # Align bars for the corresponding struct
                scores = []
                for struct in structs:
                    # Extract the score for the struct and grouping index
                    score = current_group[current_group['struct_name'] == struct]['score']
                    scores.append(score.iloc[0] if not score.empty else 0)
                
                # Plot bars with offsets
                ax.barh(y_positions + offsets[i], scores, bar_height, label=f'Group {g_idx}')

            # Label and style the chart
            ax.set_yticks(y_positions)
            ax.set_yticklabels(structs)
            ax.set_xlabel('Score')
            ax.set_ylabel('Struct Name')
            ax.set_title('Scores by Struct and Grouping Index')
            ax.legend(title='Grouping Index')
            plt.tight_layout()

            # Save the figure
            plt.savefig(self.config.experiment_dir.joinpath("dataset_struct_grouping.png"))
            # print("Figure saved to test.png")


    def _visualize_grouping_percentage(self, df: pd.DataFrame):
        """
        Generate a pie chart showing the percentage contribution of each grouping_idx
        in being the best score for the structs, with the grouper names displayed.
        Groups are displayed in ascending order of their indices.
        
        :param df: A Pandas DataFrame with columns: 
                    'struct_name', 'grouping_idx', 'score'
        """
        # Find the row with the highest score for each struct
        max_scores = df.loc[df.groupby('struct_name')['score'].idxmax()]

        # Count the occurrences of each grouping_idx
        grouping_counts = max_scores['grouping_idx'].value_counts()

        # Retrieve all grouper names
        grouper_names = groupers.get_all_grouper_names()

        # Sort by grouping index for consistent order
        grouping_counts = grouping_counts.sort_index()

        # Extract data for plotting
        grouping_indices = grouping_counts.index
        sizes = grouping_counts.values

        # Combine grouping index with their respective names
        labels = [
            f'Group {g_idx}: {grouper_names[g_idx]}' if g_idx < len(grouper_names) else f'Group {g_idx}'
            for g_idx in grouping_indices
        ]

        # Use the tab20 colormap for consistent coloring
        cmap = cm.get_cmap('tab20', len(labels))
        colors = [cmap(i) for i in range(len(labels))]

        # Plot the pie chart
        fig, ax = plt.subplots(figsize=(8, 8))
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            autopct=lambda p: f'{p:.1f}%' if p > 5 else '',  # Hide small percentages
            startangle=90,
            colors=colors,
            textprops={'fontsize': 10},
        )

        # Add a title and style the chart
        ax.set_title('Percentage of Best Scores by Grouping Index', fontsize=14)

        # Save the figure
        plt.tight_layout()
        plt.savefig(self.config.experiment_dir.joinpath("dataset_grouping_percentage_pie.png"))


    
    def _visualize_struct_best_grouping(self, df: pd.DataFrame):
        """
        Visualize scores for each struct as horizontal bar charts.
        Each struct displays only the grouping_idx with the highest score,
        with different colors representing each grouping_idx.
        
        :param df: A Pandas DataFrame with columns: 
                    'struct_name', 'grouping_idx', 'score'
        """
        # Find the row with the highest score for each struct
        max_scores = df.loc[df.groupby('struct_name')['score'].idxmax()]

        # Sort by struct_name for consistent plotting
        max_scores = max_scores.sort_values(by='struct_name')

        # Extract data for plotting
        structs = max_scores['struct_name'].values
        scores = max_scores['score'].values
        grouping_indices = max_scores['grouping_idx'].values

        # Use the tab20 colormap to assign colors based on grouping indices
        unique_groups = np.unique(grouping_indices)
        cmap = cm.get_cmap('tab20', len(unique_groups))
        colors = {g_idx: cmap(i) for i, g_idx in enumerate(unique_groups)}
        bar_colors = [colors[g_idx] for g_idx in grouping_indices]

        # Initialize the plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # Set up y-axis positions for the structs
        y_positions = np.arange(len(structs))

        # Create horizontal bars with colors based on grouping indices
        ax.barh(y_positions, scores, color=bar_colors)

        # Label and style the chart
        ax.set_yticks(y_positions)
        ax.set_yticklabels(structs)
        ax.set_xlabel('Score')
        ax.set_title('Highest Scores by Struct (Grouped by Index)')

        # Add a legend for grouping indices
        legend_labels = [f'Group {g_idx}' for g_idx in unique_groups]
        legend_handles = [plt.Rectangle((0, 0), 1, 1, color=colors[g_idx]) for g_idx in unique_groups]
        ax.legend(legend_handles, legend_labels, title='Grouping Index', bbox_to_anchor=(1.05, 1), loc='upper left')

        # Adjust layout to prevent clipping
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space for the legend

        # Save the figure
        plt.savefig(self.config.experiment_dir.joinpath("dataset_struct_grouping.png"))

    
    
    def analyze_dataset(self, ds_df=None):  
        print("Visualizing dataset") 
        if ds_df is None: 
            ds_df = pd.read_csv(self.config.analysis_ds_df_path) 
        print(ds_df.head()) 
        self._analyze_dataset(ds_df)  

    def predict_and_transform_all(self, directory=".", build_dir=Path("build"), model_path=models.DEFAULT_MODEL_PATH): 
        folder = Path(directory) 
        print(f"Transforming all benchmarks under folder {folder}")
        transform_result_dict = {} 
        total_cnt = 0 
        positive_cnt = 0
        columns = ["file", "grouping_ids_dict", "score", "time_delta", "d1_miss_rate", "lld_miss_delta"]
        transform_result_df = pd.DataFrame(columns=columns) 
        
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
                transform_result = {
                    "file": f.name, 
                    "grouping_ids_dict": grouping_ids_dict, 
                    "score" : score*100, 
                    "time_delta" : avg_time_delta*100,  
                    "d1_miss_delta": avg_d1_miss_delta*100, 
                    "lld_miss_delta" : avg_lld_miss_delta*100 # scale into percentage 
                }
                transform_result_dict[f.name] = transform_result
                transform_result_df = transform_result_df._append(transform_result, ignore_index=True) 
                total_cnt += 1 
                if score > 0: 
                    positive_cnt+=1 
        
        positive_rate = positive_cnt / total_cnt 
        print("final transform result: \n", json.dumps(transform_result_dict, indent=4)) 
        print(f"Total cnt: {total_cnt} Positive cnt: {positive_cnt} Positive rate: {positive_rate}")
        with open(self.config.experiment_dir.joinpath(TRANSFORM_RESULT_FNAME), "w") as f: 
            json.dump(transform_result_dict, f, indent=4) 

        processed_result_df = transform_result_df.copy() 
        processed_result_df["grouping_ids_dict"] = processed_result_df["grouping_ids_dict"].apply(lambda x: json.dumps(x)) 
        processed_result_df.to_csv(self.config.transform_result_path, index=False) 
        self.analyze_and_generate_bar_charts(transform_result_df, output_dir=self.config.experiment_dir)

    def get_model_path(self): 
        if self.config.model_path is None: 
            return self.get_experiment_dir().joinpath("model.pth") 
        return self.config.model_path
    
    def transform_all(self):
        self.predict_and_transform_all(directory=self.config.benchmark_dir, build_dir=self.config.build_dir, model_path=self.get_model_path())

    def analyze_and_transform_all(self):
        print("Analyze and transform all")
        # df, ds_df = self.analyze_all_benchmarks(directory=self.config.benchmark_dir, build_dir=self.config.build_dir)
        ds_df = self.get_all_benchmark_ds_df() 
        self.train_selector(ds_df=ds_df, epochs=self.config.train_epochs)
        self.predict_and_transform_all(directory=self.config.benchmark_dir, build_dir=self.config.build_dir, model_path=self.get_model_path())

    def predict_transform(self):
        source_file = Path(self.config.source_file)
        print(f"Predict transform for source file: {source_file}")
        self.analyzer.run_setup()
        self.analyzer.load_analysis_file()
        self.analyzer.make_prediction(self.get_model_path())

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
            analyzer = analyze.StructAnalyzer(config=self.config) 
            analyzer.run_setup() 
            analyzer.load_analysis_file()
            analyzer.sanity_check()
    
    def sanity_check_all(self):
        print(f"Running sanity check for benchmark directory: {self.config.benchmark_dir}")
        self._sanity_check_all(self.config.benchmark_dir, self.config.build_dir)
    
    def _evaluate_model(self, ds_df:pd.DataFrame, model_path:Path=Path(models.DEFAULT_MODEL_PATH)): 
        selector_model = models.GroupingSelector(C=groupers.GROUPERS_CNT,L=self.config.loop_cnt,  hidden_size=self.config.hidden_size)  
        selector_model.load_model(model_path) 
        train_df, test_df = train_test_split(ds_df, test_size=0.2, train_size=0.8) 
        # selector_model.evaluate(ds_df)
        selector_model.evaluate(test_df)  
    
    def evaluate_model(self): 
        print(f"Evaluating model on all benchmarks") 
        # df, ds_df = self.analyze_all_benchmarks(directory=self.config.benchmark_dir, build_dir=self.config.build_dir)
        ds_df = self.get_all_benchmark_ds_df() 
        self._evaluate_model(ds_df, self.get_model_path()) 

    def run(self):
        start = time.time() 
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

        if self.config.analyze_transform_result: 
            self.analyze_transform_result() 
            
        if self.config.analyze_dataset: 
            self.analyze_dataset() 
            
        duration = time.time() - start 
        print(f"Run took {duration} seconds!") 
