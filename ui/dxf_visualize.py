import ezdxf
import sys, os, copy
import numpy as np  # Import numpy for calculations
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Arc

class DXFVisualize:
    def __init__(self):
        pass

    # Function to draw a circle
    def draw_circle(self, ax, center, radius):
        circle = Circle(center, radius, fill=False, color='black')
        ax.add_patch(circle)

    def draw_polyline(self, ax, vertices):
        latitudes = []
        longitudes = []
        points = []

        for line in vertices:

            lat = line[0][0]
            long = line[0][1]

            latitudes.append(lat)
            longitudes.append(long)
            points.append([lat, long])

        ax.plot(latitudes, longitudes, marker='o')

    # Function to draw a lightweight polyline (LWPolyline)
    def draw_lwpolyline(self, ax, points, is_closed):
        if is_closed:
            polygon = Polygon(points, closed=True, fill=False, edgecolor='black')
            ax.add_patch(polygon)
        else:
            ax.plot([p[0] for p in points], [p[1] for p in points], 'k-')

    def draw_point(self, ax, location):
        """
        Draw a point on the matplotlib axis.

        Args:
            ax: Matplotlib axis to draw on.
            location: Tuple containing the (x, y) coordinates of the point.
        """
        # Draw a point using matplotlib scatter plot
        ax.scatter(location[0], location[1], color='black', marker='o')

    def draw_text(self, ax, position, text_content):
        if "Piece Name:" in text_content:
            y_offset = -400
            ax.text(position[0] - 500, position[1] + y_offset, text_content, color="red")
        elif "Size:" in text_content:
            y_offset = -420
            ax.text(position[0] - 500, position[1] + y_offset, text_content, color="red")
        elif "Annotation:" in text_content:
            y_offset = -440
            ax.text(position[0] - 500, position[1] + y_offset, text_content, color="red")
        elif "Quantity:" in text_content:
            y_offset = -460
            ax.text(position[0] - 500, position[1] + y_offset, text_content, color="red")
        elif "Category:" in text_content:
            y_offset = -480
            ax.text(position[0] - 500, position[1] + y_offset, text_content, color="red")
        else:
            ax.text(position[0], position[1], text_content, color="black")


    # Function to draw a line
    def draw_line(self, ax, start, end):
        ax.plot([start[0], end[0]], [start[1], end[1]], 'k-')

    def add_arc(self, ax, start, end, bulge):
        # Calculate the arc's radius and center
        dx, dy = end[0] - start[0], end[1] - start[1]
        dist = np.sqrt(dx**2 + dy**2)
        radius = dist * (1 + bulge**2) / (2 * bulge)
        
        # Middle point between start and end
        mid = [(start[0] + end[0]) / 2, (start[1] + end[1]) / 2]
        
        # Distance from midpoint to arc center
        sagitta = radius - dist / 2 * abs(bulge)
        angle = np.arctan2(dy, dx)
        
        # Determine center of the arc
        if bulge > 0:
            center = [mid[0] + sagitta * np.sin(angle), mid[1] - sagitta * np.cos(angle)]
        else:
            center = [mid[0] - sagitta * np.sin(angle), mid[1] + sagitta * np.cos(angle)]
        
        # Calculate start and end angles
        start_angle = np.degrees(np.arctan2(start[1] - center[1], start[0] - center[0]))
        end_angle = np.degrees(np.arctan2(end[1] - center[1], end[0] - center[0]))
        
        # Arc drawing
        if bulge < 0:
            if start_angle < end_angle:
                start_angle += 360
        else:
            if end_angle < start_angle:
                end_angle += 360

        arc = Arc(center, 2*radius, 2*radius, angle=0, theta1=start_angle, theta2=end_angle, color='black')
        ax.add_patch(arc)

    def draw_lwpolyline_o(self, ax, entity):
        vertices = entity.get_points(format='xyb')
        for i in range(len(vertices) - 1):
            start, end = vertices[i], vertices[i + 1]
            bulge = start[2]
            if bulge == 0:
                ax.plot([start[0], end[0]], [start[1], end[1]], 'k-')
            else:
                self.add_arc(ax, start, end, bulge)

if __name__ == "__main__":
    dxf_visualize = DXFVisualize()

    if len(sys.argv) < 3:
        print("Usage: {} input.dxf output.png".format(sys.argv[0]))
        sys.exit(1)

    infilename = sys.argv[1]
    outfilename = sys.argv[2]

    if os.path.exists(outfilename):
        sys.exit("Output file {} exists".format(outfilename))

    # Read the DXF file
    doc = ezdxf.readfile(infilename)
    msp = doc.modelspace()

    # Prepare a matplotlib figure
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    # Iterate through entities in the model space
    for entity in msp:
        if entity.dxftype() == 'LINE':
            dxf_visualize.draw_line(ax, entity.dxf.start, entity.dxf.end)
        elif entity.dxftype() == 'CIRCLE':
            dxf_visualize.draw_circle(ax, (-entity.dxf.center.x, entity.dxf.center.y), entity.dxf.radius)
        elif entity.dxftype() == 'LWPOLYLINE':
            dxf_visualize.draw_lwpolyline(ax, entity)
    
    # Set aspect ratio and limits for better visualization
    ax.autoscale_view()
    
    # Save the figure to a PNG file
    plt.savefig(outfilename, dpi=300)
    
    # Optionally, display the plot
    plt.show()
