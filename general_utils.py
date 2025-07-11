import os
import json
from dotenv import load_dotenv
import shutil
import re

def clean_folder(folder_path):
    """
    Deletes all files and subdirectories except .gitkeep within the specified folder.

    Args:
        folder_path (str): The path to the folder to be cleaned.

    Raises:
        ValueError: If the provided path is not a valid directory.
        Exception: Prints an error message if a file or folder cannot be deleted.
    """
    if not os.path.isdir(folder_path):
        raise ValueError(f"{folder_path} is not a valid directory")

    for item in os.listdir(folder_path):
        if item == '.gitkeep':
            continue
        item_path = os.path.join(folder_path, item)

        try:            
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.remove(item_path)  # Remove files or symbolic links
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)  # Remove entire subdirectories
        except Exception as e:
            print(f"Failed to delete {item_path}: {e}")

# Load environment variables from a .env file into the system environment
load_dotenv()

def load_json_list(env_var):
    """
    Loads a JSON-formatted list from an environment variable.

    Args:
        env_var (str): The name of the environment variable.

    Returns:
        list: The JSON-decoded list from the environment variable.

    Raises:
        ValueError: If the environment variable is not set, is empty, 
                    or contains invalid JSON.
    """
    val = os.getenv(env_var)
    if not val:
        raise ValueError(f"Environment variable '{env_var}' is not set or is empty.")
    try:
        return json.loads(val)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in environment variable '{env_var}': {e}")

def clean_filename_keep_ext(filename):
    """
    Cleans the filename by removing the last underscore-separated part (if there are two or more underscores),
    while preserving the file extension.

    Example:
        'file_name_part.txt' -> 'file_name.txt'

    Args:
        filename (str): The original filename.

    Returns:
        str: The cleaned filename with the original extension.
    """
    name, ext = os.path.splitext(filename)
    if name.count('_') >= 2:
        name = re.sub(r'_[^_]+$', '', name)  # Remove the last underscore and following part
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