import os
import ezdxf
import sys

class DXFtoSVGConverter:
    def __init__(self, canvas_size):
        self.canvas_size = canvas_size
        self.doc = None
        self.msp = None

    def load_dxf(self, dxf_file_path):
        self.file_path = dxf_file_path
        try:
            self.doc = ezdxf.readfile(self.file_path)
            self.msp = self.doc.modelspace()  # the common construction space
        except (IOError, ezdxf.DXFStructureError) as e:
            print(f"Error loading DXF file '{self.file_path}': {e}")
            sys.exit(1)

    def convert_to_svg(self, dxf_file_path, output_svg_path):
        self.load_dxf(dxf_file_path)
        min_x, min_y, max_x, max_y = self.calculate_bounding_box()

        if min_x == float('inf') or min_y == float('inf') or max_x == float('-inf') or max_y == float('-inf'):
            print(f"No valid entities found in the DXF file '{dxf_file_path}'.")
            return

        # Calculate scaling factors to fit the DXF content into the specified canvas size
        canvas_width, canvas_height = self.canvas_size
        scale_factor = min(canvas_width / (max_x - min_x), canvas_height / (max_y - min_y))

        # Ensure the output file has the correct extension
        svg_filename = os.path.splitext(os.path.basename(dxf_file_path))[0] + '.svg'
        svg_filepath = os.path.join(output_svg_path, svg_filename)
        os.makedirs(os.path.dirname(svg_filepath), exist_ok=True)

        with open(svg_filepath, "w") as svg_file:
            # Write SVG header with specified canvas size
            svg_file.write(f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="{canvas_width}" height="{canvas_height}">\n')
            # Apply scaling transformation
            svg_file.write(f'<g transform="scale({scale_factor}) translate({-min_x}, {-min_y})">\n')

            for entity in self.msp:
                self.write_svg_entity(svg_file, entity)

            # Close scaling transformation group
            svg_file.write('</g>\n')
            # Close SVG file
            svg_file.write('</svg>')

        print(f"DXF file converted to SVG: {svg_filepath}")

    def calculate_bounding_box(self):
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')

        for entity in self.msp:
            if entity.dxftype() == 'LINE':
                start = entity.dxf.start
                end = entity.dxf.end
                min_x, min_y = min(min_x, start.x, end.x), min(min_y, start.y, end.y)
                max_x, max_y = max(max_x, start.x, end.x), max(max_y, start.y, end.y)
            elif entity.dxftype() == 'CIRCLE':
                center = entity.dxf.center
                radius = entity.dxf.radius
                min_x, min_y = min(min_x, center.x - radius), min(min_y, center.y - radius)
                max_x, max_y = max(max_x, center.x + radius), max(max_y, center.y + radius)
            elif entity.dxftype() in ('POLYLINE', 'LWPOLYLINE'):
                if entity.dxftype() == 'POLYLINE':
                    vertices = [(vertex.dxf.location.x, vertex.dxf.location.y) for vertex in entity.vertices]
                else:
                    vertices = entity.get_points('xy')
                for x, y in vertices:
                    min_x, min_y = min(min_x, x), min(min_y, y)
                    max_x, max_y = max(max_x, x), max(max_y, y)

        return min_x, min_y, max_x, max_y

    def write_svg_entity(self, svg_file, entity):
        if entity.dxftype() == 'LINE':
            start = entity.dxf.start
            end = entity.dxf.end
            svg_file.write(f'<line x1="{start.x}" y1="{start.y}" x2="{end.x}" y2="{end.y}" stroke="black" />\n')
            # Annotate vertices
            svg_file.write(f'<text x="{start.x}" y="{start.y}" fill="red" font-size="2">1</text>\n')
            svg_file.write(f'<text x="{end.x}" y="{end.y}" fill="red" font-size="2">2</text>\n')
        elif entity.dxftype() == 'CIRCLE':
            center = entity.dxf.center
            radius = entity.dxf.radius
            svg_file.write(f'<circle cx="{center.x}" cy="{center.y}" r="{radius}" stroke="black" fill="none" />\n')
        elif entity.dxftype() in ('POLYLINE', 'LWPOLYLINE'):
            if entity.dxftype() == 'POLYLINE':
                points = [(vertex.dxf.location.x, vertex.dxf.location.y) for vertex in entity.vertices]
            else:
                points = entity.get_points('xy')
            if entity.is_closed:
                svg_file.write('<polygon points="')
            else:
                svg_file.write('<polyline points="')
            for x, y in points:
                svg_file.write(f'{x},{y} ')
            svg_file.write('" stroke="black" fill="none" />\n')
            # Annotate vertices
            for i, (x, y) in enumerate(points):
                svg_file.write(f'<text x="{x}" y="{y}" fill="red" font-size="2">{i + 1}</text>\n')

if __name__ == "__main__":
    pattern = "basic_pattern/"
    #pattern = ""

    dxf_directory = "../data/ff_pattern_2/" + str(pattern)
    svg_output_directory = "../data/output_svg/" + pattern

    canvas_size = (2480, 3508)  # A4 format

    converter = DXFtoSVGConverter(canvas_size)

    for filename in os.listdir(dxf_directory):
        file_path = os.path.join(dxf_directory, filename)
        print(f"Processing file: {file_path}")
        if os.path.isfile(file_path) and filename.lower().endswith('.dxf'):
            converter.convert_to_svg(file_path, svg_output_directory)
            print("done")
