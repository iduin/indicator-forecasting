import os
import json
from dotenv import load_dotenv

load_dotenv()

def load_json_list(env_var):
    val = os.getenv(env_var)
    return json.loads(val) if val else []

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