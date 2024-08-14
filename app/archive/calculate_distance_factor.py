def convert_coordinates_to_mm(coords, distances_mm):
    # Calculate the differences in coordinates
    delta_x_8000_8022 = coords["8022"][0] - coords["8000"][0]
    delta_y_8000_8001 = coords["8001"][1] - coords["8000"][1]
    delta_x_8014_8017 = coords["8017"][0] - coords["8014"][0]
    delta_y_8014_8017 = coords["8017"][1] - coords["8014"][1]

    # Calculate the conversion factors
    conversion_factor_x = distances_mm["8000-8022-X"] / delta_x_8000_8022
    conversion_factor_y = distances_mm["8000-8001-Y"] / delta_y_8000_8001

    # Verify if the conversion factors match for other distances (optional)
    conversion_factor_x_check = distances_mm["8014-8017-X"] / delta_x_8014_8017
    conversion_factor_y_check = distances_mm["8014-8017-Y"] / delta_y_8014_8017

    # Print the conversion factors
    print(f"Conversion Factor X: {conversion_factor_x}")
    print(f"Conversion Factor Y: {conversion_factor_y}")
    print(f"Conversion Factor X (Check): {conversion_factor_x_check}")
    print(f"Conversion Factor Y (Check): {conversion_factor_y_check}")

    # Convert all coordinates to mm
    coords_mm = {key: (x * conversion_factor_x, y * conversion_factor_y) for key, (x, y) in coords.items()}

    return coords_mm

# Example usage
coords = {
    "8000": (0.86, 0.0),
    "8001": (0.86, 2.5),
    "8022": (30.41, 0.0),
    "8014": (9.74, 14.95),
    "8017": (28.34, 14.55)
}

distances_mm = {
    "8000-8022-X": 750.64,
    "8000-8001-Y": 63.50,
    "8014-8017-X": 472.313,
    "8014-8017-Y": -10.16
}

# Call the function
converted_coords = convert_coordinates_to_mm(coords, distances_mm)

# Print the converted coordinates
print("Converted Coordinates (in mm):")
for key, (x, y) in converted_coords.items():
    print(f"{key}: ({x:.2f} mm, {y:.2f} mm)")
