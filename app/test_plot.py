import matplotlib.pyplot as plt

# Given coordinates
new_xs = (30.41, 0.857, 0.857, 2.455, 2.325, 1.899, 0.91, 0.0, 2.619, 4.408, 6.99, 8.127, 8.743, 9.419, 26.639, 28.335, 28.546, 28.979, 29.628, 30.063, 30.359, 30.41, 30.41)
new_ys = (0.0, 0.0, 2.5, 2.5, 3.692, 4.53, 5.296, 5.572, 11.948, 11.673, 11.512, 11.775, 12.19, 13.177, 14.522, 14.552, 13.218, 12.024, 10.766, 9.547, 7.501, 2.5, 0.0)

# Plot the points in the order they are provided
plt.figure(figsize=(10, 6))
plt.plot(new_xs, new_ys, marker='o', linewidth=1.5, markersize=7, linestyle='-', color='blue')

# Add labels to the points for better visualization
for i, (x, y) in enumerate(zip(new_xs, new_ys)):
    plt.text(x, y, f'({x}, {y})', fontsize=8)

plt.title('Polygon with Given Coordinates')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(True)
plt.show()
