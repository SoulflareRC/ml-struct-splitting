import sys 
from runner import Runner 
from tqdm import tqdm 
def auto_experiments(experiments:list):  
    '''
    takes a list of dict 
    '''
    for idx, e in tqdm(enumerate(experiments)): 
        print(f"RUNNING EXPERIMENT {idx}") 
        runner = Runner(config_override=e) 
        runner.run() 
if __name__=="__main__": 
    experiments = [
        # {
        #     "experiment_name":"base", 
        #     "analyze_and_transform_all": True, 
        # }, 
        # {
        #     "experiment_name":"base-BB", 
        #     "feature_mode": "BB",  
        #     "analyze_and_transform_all": True, 
        # }, 
        # {
        #     "experiment_name":"base-L5", 
        #     "loop_cnt": 5,  
        #     "analyze_and_transform_all": True, 
        # }, 
        # {
        #     "experiment_name":"base-LSTM", 
        #     "rnn_type": "LSTM",  
        #     "analyze_and_transform_all": True, 
        # }, 
        # {
        #     "experiment_name":"base-RNN", 
        #     "rnn_type": "RNN",  
        #     "analyze_and_transform_all": True, 
        # }, 
        # {
        #     "experiment_name":"base-H8", 
        #     "hidden_size": 8,  
        #     "analyze_and_transform_all": True, 
        # }, 
        {
            "experiment_name":"base-maxN-5", 
            "max_N": 5,  
            "analyze_and_transform_all": True, 
        }, 
    ]
    auto_experiments(experiments=experiments) 