import os
import json




def remove_last_character_if_bracket(file_path):
    with open(file_path, 'rb+') as file:
        file.seek(-1, os.SEEK_END)
        if file.read(1) == b"]":
            file.seek(-1, os.SEEK_END)
            file.truncate()

def append_to_json_list(data_record, file_path):
    with open(file_path, 'a') as file:
        # Check if the file already contains data and hence, requires a comma
        file.seek(0, os.SEEK_END)  # Move the cursor to the end of the file
        position = file.tell()  # Get the current position
        if position > 1:  # More than just the opening bracket
            file.write(",")  # Add a comma to separate the new record
        json.dump(data_record, file)  # Write the new data record

def finalize_json_list(file_path):
    with open(file_path, 'a') as file:
        file.write("]")

def save_to_file(args: dict, data_record: list, file_name: str):
    path_name = f"GPT_Prompts/agent_tuning_vicuna-13b-v1.5/{args.base_model}/"
    file_name = f"{file_name}.json"
    file_path = os.path.join(path_name, file_name)

    if not os.path.exists(path_name):
        os.makedirs(path_name)

    # If the file doesn't exist, create it and write an opening bracket.
    # If it exists, remove the last character if it's a closing bracket.
    if not os.path.exists(file_path):
        with open(file_path, 'w') as file:
            file.write("[")
    else:
        remove_last_character_if_bracket(file_path)

    # Append the data_record list to the file
    #for record in data_record:
    append_to_json_list(data_record, file_path)

    # After appending all records, finalize the JSON list with a closing bracket
    finalize_json_list(file_path)