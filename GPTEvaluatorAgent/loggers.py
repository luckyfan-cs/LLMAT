import os

import json




def save_to_file(args: str, GPT_instructions):
    path_name = "GPT_Prompts/" + args.base_model + "/"
    file_name =  args.file_name
    file_name = file_name + ".json"
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    with open(path_name + file_name, 'w') as file:
        json.dump(GPT_instructions, file)
    

