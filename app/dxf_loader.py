import os
import ezdxf
import sys
from ezdxf.groupby import groupby
from dxf_visualize import DXFVisualize
import matplotlib.pyplot as plt


class DXFLoader:
    def __init__(self):
        self.file_path = None
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

    def process_text_entities(self):
        # Iterate through entities in the model space
        for entity in self.msp:
            if entity.dxftype() == "TEXT":
                text_content = entity.dxf.text
                position = (entity.dxf.insert[0], entity.dxf.insert[1])  # Text position (x, y)
                height = entity.dxf.height  # Text height
                style = entity.dxf.style  # Text style
                # Process or store the text information as needed
                # print(f"Text: {text_content}, Position: {position}, Height: {height}, Style: {style}")

    def group_by_layer(self):
        return groupby(entities=self.msp, dxfattrib="layer")

    def group_by_custom(self, input_key):
        return self.msp.groupby(key=input_key)

    @staticmethod
    def layer_and_color_key(entity):
        return (entity.dxf.layer, entity.dxf.color) if entity.dxf.layer != "0" else None

    def group_by_color(self):
        grouped_entities = {}
        for entity in self.msp:
            color = entity.dxf.color
            grouped_entities.setdefault(color, []).append(entity)
        return grouped_entities

    def group_by_entity_type(self):
        grouped_entities = {}
        for entity in self.msp:
            entity_type = entity.dxftype()
            grouped_entities.setdefault(entity_type, []).append(entity)
        return grouped_entities

    def group_by_blocks(self):
        grouped_blocks = {}
        for entity in self.msp:
            if entity.dxftype() == 'INSERT':
                block_name = entity.dxf.name
                grouped_blocks.setdefault(block_name, []).append(entity)
        return grouped_blocks

    def view_block_geometry(self, block_name):
        for entity in self.msp.query(f'INSERT[name=="{block_name}"]'):
            block = self.doc.blocks.get(entity.dxf.name)
            if block is None:
                print(f"Block '{entity.dxf.name}' not found.")
                continue

            print(f"Geometry of block '{entity.dxf.name}':")
            for e in block:
                print(f"Type: {e.dxftype()}")
                for attribute in e.dxf.all_existing_dxf_attribs():
                    print(f"  {attribute}: {getattr(e.dxf, attribute)}")

    def visualize_block(self, block_name, output_folder):
        dxf_visualize = DXFVisualize()
        fig, ax = plt.subplots()
        ax.set_aspect('equal')

        block_entities = self.msp.query(f'INSERT[name=="{block_name}"]')
        if not block_entities:
            print(f"Block '{block_name}' not found in the model space.")
            return

        for insert_entity in block_entities:
            block = self.doc.blocks.get(insert_entity.dxf.name)
            if block is None:
                print(f"Block '{insert_entity.dxf.name}' not found.")
                continue

            for entity in block:
                if entity.dxftype() == 'LINE':
                    start = (entity.dxf.start[0], entity.dxf.start[1])
                    end = (entity.dxf.end[0], entity.dxf.end[1])
                    dxf_visualize.draw_line(ax, start, end)
                elif entity.dxftype() == 'CIRCLE':
                    dxf_visualize.draw_circle(ax, (-entity.dxf.center.x, entity.dxf.center.y), entity.dxf.radius)
                elif entity.dxftype() == 'POLYLINE':
                    vertices = [(vertex.dxf.location[0], vertex.dxf.location[1]) for vertex in entity.vertices()]
                    dxf_visualize.draw_polyline(ax, vertices)
                elif entity.dxftype() == 'TEXT':
                    position = (entity.dxf.insert[0], entity.dxf.insert[1]) if isinstance(entity.dxf.insert, tuple) else (
                        entity.dxf.align_point[0], entity.dxf.align_point[1])
                    text_content = entity.dxf.text
                    dxf_visualize.draw_text(ax, position, text_content)

        ax.autoscale_view()
        output_path = os.path.join(output_folder, block_name + ".png")
        plt.savefig(output_path, dpi=300)
        plt.close()


if __name__ == "__main__":
    dxf_directory = "../data/dxf/"
    output_image_directory = "../data/output_images"

    dxf_loader = DXFLoader()

    for filename in os.listdir(dxf_directory):
        file_path = os.path.join(dxf_directory, filename)
        if os.path.isfile(file_path) and filename.lower().endswith('.dxf'):
            dxf_loader.load_dxf(file_path)
            dxf_loader.process_text_entities()
            dxf_loader.group_by_layer()
            dxf_loader.group_by_custom(DXFLoader.layer_and_color_key)
            dxf_loader.group_by_entity_type()
            dxf_loader.group_by_blocks()
            dxf_loader.group_by_color()

            for block_name in dxf_loader.group_by_blocks().keys():
                print(f'Block: {block_name}')
                dxf_loader.view_block_geometry(block_name)
                print("-" * 40)
