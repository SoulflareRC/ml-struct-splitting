import json 
class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __repr__(self):
        return json.dumps(self.__dict__, indent=4)
    
    def load_config(self, config_dict):
        """
        Update the current configuration instance with values from a dictionary.
        """
        for key, value in config_dict.items():
            setattr(self, key, value)
    
    def save_config(self):
        """
        Save the current configuration to a dictionary.
        """
        return {key: getattr(self, key) for key in self.__dict__}