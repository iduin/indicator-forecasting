import os
import json
from dotenv import load_dotenv
import shutil
import re

def clean_folder(folder_path):
    if not os.path.isdir(folder_path):
        raise ValueError(f"{folder_path} is not a valid directory")

    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)

        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        except Exception as e:
            print(f"Failed to delete {item_path}: {e}")

load_dotenv()

def load_json_list(env_var):
    val = os.getenv(env_var)
    if not val:
        raise ValueError(f"Environment variable '{env_var}' is not set or is empty.")
    try:
        return json.loads(val)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in environment variable '{env_var}': {e}")
    

def clean_filename_keep_ext(filename):
    name, ext = os.path.splitext(filename)
    if name.count('_') >= 2:
        name = re.sub(r'_[^_]+$', '', name)
    return f"{name}{ext}"


if __name__ == '__main__' :
    DATA_DIR = os.getenv("DATA_DIR")

    INDICS = load_json_list("INDICS")
    TRAIN_SHEETS = load_json_list("TRAIN_SHEETS")
    TEST_SHEETS = load_json_list("TEST_SHEETS")
    TRAIN_SYNTH_SHEETS = load_json_list("TRAIN_SYNTH_SHEETS")
    TEST_SYNTH_SHEETS = load_json_list("TEST_SYNTH_SHEETS")

    print(DATA_DIR)
    print(INDICS)
    print(TRAIN_SHEETS)
    print(TEST_SHEETS)
    print(TRAIN_SYNTH_SHEETS)
    print(TEST_SYNTH_SHEETS)