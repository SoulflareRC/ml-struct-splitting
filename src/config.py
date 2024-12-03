from pathlib import Path
import json

class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __repr__(self):
        return json.dumps(self.__dict__, indent=4, default=lambda x: str(x)) 
    
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
    
    def save_to_file(self, file_path: str):
        """
        Save the current configuration as a JSON file to the specified path.
        
        Parameters:
        - file_path (str): The file path to save the configuration JSON.
        """
        file_path = Path(file_path)
        try:
            with file_path.open("w", encoding="utf-8") as file:
                json.dump(self.__dict__, file, indent=4, default=lambda x: str(x))
            print(f"Configuration saved to {file_path}")
        except Exception as e:
            print(f"Failed to save configuration: {e}")
    
    def update_from_dict(self, update_dict, override_keys=[]):
        """
        Update the configuration instance with key-value pairs from a dictionary.
        
        Parameters:
        - update_dict (dict): A dictionary containing key-value pairs to update.
        """
        for key, value in update_dict.items():
            if key in override_keys or len(override_keys)==0: 
                print(f"Overriding: {key}:{value}")
                setattr(self, key, value)
        print(self.__dict__)