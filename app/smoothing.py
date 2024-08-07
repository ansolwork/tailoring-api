import numpy as np
from scipy.interpolate import Rbf, splprep, splev

class SmoothingFunctions:
    def __init__(self, vertices, start_index, end_index, reverse):
        self.vertices = vertices
        self.start_index = start_index
        self.end_index = end_index 
        self.reverse = reverse

    def apply_rbf_smoothing(self, num_points=100, ascending=None, function='multiquadric', epsilon=None):
        if len(self.vertices) < 3:
            return self.vertices

        points = np.array(self.vertices)
        if self.reverse:
            points = points[::-1]

        try:
            x, y = points[:, 0], points[:, 1]
            if epsilon is None:
                epsilon = np.mean(np.diff(x)) * 10

            rbf = Rbf(x, y, function=function, epsilon=epsilon)

            x_new = np.linspace(x[0], x[-1], num_points)
            y_new = rbf(x_new)

            smoothed_points = list(zip(x_new, y_new))

            smoothed_points[0] = tuple(self.vertices[0])
            smoothed_points[-1] = tuple(self.vertices[-1])

            if self.reverse:
                smoothed_points = smoothed_points[::-1]

            if ascending is not None:
                smoothed_points = sorted(smoothed_points, key=lambda p: (p[0], p[1]) if ascending else (-p[0], -p[1]))

        except Exception as e:
            print(f"Error applying RBF smoothing: {e}")
            smoothed_points = self.vertices

        return smoothed_points

    def apply_bezier_smoothing(self, num_points=100, ascending=None):
        if len(self.vertices) < 3:
            return self.vertices

        points = np.array(self.vertices)
        if self.reverse:
            points = points[::-1]

        try:
            tck, u = splprep([points[:, 0], points[:, 1]], s=0, k=min(1, len(points)-1))
            unew = np.linspace(0, 1, num_points)
            out = splev(unew, tck)
            smoothed_points = list(zip(out[0], out[1]))

            smoothed_points[0] = tuple(self.vertices[0])
            smoothed_points[-1] = tuple(self.vertices[-1])

            if self.reverse:
                smoothed_points = smoothed_points[::-1]

            if ascending is not None:
                smoothed_points = sorted(smoothed_points, key=lambda p: (p[0], p[1]) if ascending else (-p[0], -p[1]))

        except Exception as e:
            print(f"Error applying bezier smoothing: {e}")
            smoothed_points = self.vertices

        return smoothed_points

    def apply_linear_smoothing(self, vertices, shift, ascending, change_x, change_y):
        num_points = abs(self.end_index - self.start_index) + 1
        print(f"Debug: start_index={self.start_index}, end_index={self.end_index}, num_points={num_points}, vertices_length={len(vertices)}")
        
        if num_points > len(vertices):
            raise ValueError(f"num_points ({num_points}) is greater than the length of vertices ({len(vertices)})")

        x_shift = np.zeros(num_points)
        y_shift = np.zeros(num_points)
        if change_x > 0 and change_y == 0:
            x_shift = np.linspace(0, shift[0], num_points) if ascending else np.linspace(shift[0], 0, num_points)
        elif change_y > 0 and change_x == 0:
            y_shift = np.linspace(0, shift[1], num_points) if ascending else np.linspace(shift[1], 0, num_points)
        elif change_x > 0 and change_y > 0:
            x_shift = np.linspace(0, shift[0], num_points) if ascending else np.linspace(shift[0], 0, num_points)
            y_shift = np.linspace(0, shift[1], num_points) if ascending else np.linspace(shift[1], 0, num_points)
        if self.reverse:
            x_shift = x_shift[::-1]
            y_shift = y_shift[::-1]
        for i in range(num_points):
            index = self.start_index + i if self.start_index < self.end_index else self.start_index - i
            vertices[index] = (vertices[index][0] + x_shift[i], vertices[index][1] + y_shift[i])
        return vertices

    def apply_smoothing(self, method='linear', num_points=100, ascending=None, function='multiquadric', epsilon=None, **kwargs):
        if method == 'linear':
            return self.apply_linear_smoothing(ascending=ascending, **kwargs)
        elif method == 'rbf':
            return self.apply_rbf_smoothing(num_points, ascending, function, epsilon)
        elif method == 'bezier':
            return self.apply_bezier_smoothing(num_points, ascending)
        else:
            raise ValueError(f"Unknown smoothing method: {method}")

if __name__ == "__main__":
    vertices = [(0, 0), (1, 2), (2, 3), (3, 4)]
    smoothing = SmoothingFunctions(vertices, 0, len(vertices) - 1, reverse=False)

    smoothed_vertices_linear = smoothing.apply_smoothing(method='linear', vertices=vertices, shift=(0, 0), ascending=True, change_x=1, change_y=1)
    print("Linear Smoothed vertices:", smoothed_vertices_linear)

    smoothed_vertices_rbf = smoothing.apply_smoothing(method='rbf')
    print("RBF Smoothed vertices:", smoothed_vertices_rbf)

    smoothed_vertices_bezier = smoothing.apply_smoothing(method='bezier')
    print("Bezier Smoothed vertices:", smoothed_vertices_bezier)
