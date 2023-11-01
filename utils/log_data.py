import json
import csv

from paths import DATA_PATH

CSV_SCHEMA = ["step", "robot0_eef_pos", "cube_pos", "gripper_to_cube_pos", "action", "grasp", "reward"]

def log_data(data, filename):
    data_json = json.dumps(data)
    with open(DATA_PATH + "/" + filename, "a") as f:
        f.write(data_json + "\n")

def log_csv_data(data, filename):
    with open(DATA_PATH + "/" + filename, "a") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(data)
