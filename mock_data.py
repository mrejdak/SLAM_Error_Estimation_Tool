import json
import numpy as np
import csv


def load_data_measurements():
    with open("real_data/cone_positions_reduced.json", "r") as f:
        data = np.array(json.load(f))
        positions = np.array(data[:,0:2])
        colors = np.array(data[:,2])
        return positions, colors


def load_data_csv():
    positions = []
    is_yellow = []

    with open("real_data/slam_cones_2.csv", newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if not row:
                continue  # skip empty lines

            if len(row) != 3:
                continue  # skip malformed lines

            try:
                x = float(row[0])
                y = float(row[1])
                cone_type = row[2].strip().upper()
            except ValueError:
                continue  # skip if conversion fails

            positions.append([x, y])
            is_yellow.append("YELLOW" in cone_type)

    positions = np.array(positions) * 100.0
    is_yellow = np.array(is_yellow, dtype=bool)

    return positions, is_yellow



def transform_point(point, translation, rotation_angle, old_origin):
    """
    Transform a point: translate to origin, rotate, then translate to new position.
    (Not used anymore - keeping for potential future use)
    """
    point = np.array(point)

    # Translate to origin (relative to old blue cone)
    relative = point - old_origin

    # Rotate
    cos_a = np.cos(rotation_angle)
    sin_a = np.sin(rotation_angle)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated = rotation_matrix @ relative

    # Translate to new position
    transformed = rotated + translation

    return transformed

def mock_test_data(noise=False):
    positions, colors = load_data_measurements()
    translation = [31.2, 103.2]
    rotation_angle = np.deg2rad(21)
    new_positions = []
    for (x, y) in positions:
        new_translation = translation + np.random.uniform(high=30, size=2) if noise else translation
        new_point = transform_point((x, y), new_translation, rotation_angle, np.array((0., 0.)))
        new_positions.append(new_point)
    return np.array(new_positions[::-1]), np.array(colors[::-1])
