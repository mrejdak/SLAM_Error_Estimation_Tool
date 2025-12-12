import csv
import json


def parse_json_to_csv(filename: str):
    csv_filename: str = filename.removesuffix(".json") + ".csv"
    with open(filename, 'r') as f:
        data: list[float, float, bool] = json.load(f)
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "cone_type"])
        for cone in data:
            row = [cone[0], cone[1], "CONE_YELLOW_RIGHT" if cone[2] else "CONE_BLUE_LEFT"]
            writer.writerow(row)

parse_json_to_csv("../measurementsv1/cone_positions.json")