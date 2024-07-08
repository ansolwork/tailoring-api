import os
import ezdxf
import sys
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon

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

    def extract_annotations(self):
        annotations = []
        for entity in self.msp:
            if entity.dxftype() == "TEXT":
                annotations.append({
                    "Text": entity.dxf.text,
                    "Position": (entity.dxf.insert[0], entity.dxf.insert[1])
                })
        return annotations

    def classify_annotations(self, annotations):
        classified_annotations = {
            "Piece_Name": [],
            "Size": [],
            "Annotation": [],
            "Quantity": [],
            "Category": []
        }
        for annotation in annotations:
            text = annotation["Text"]
            position = annotation["Position"]
            if "Piece Name:" in text:
                classified_annotations["Piece_Name"].append({"Text": text, "Position": position})
            elif "Size:" in text:
                classified_annotations["Size"].append({"Text": text, "Position": position})
            elif "Annotation:" in text:
                classified_annotations["Annotation"].append({"Text": text, "Position": position})
            elif "Quantity:" in text:
                classified_annotations["Quantity"].append({"Text": text, "Position": position})
            elif "Category:" in text:
                classified_annotations["Category"].append({"Text": text, "Position": position})
        return classified_annotations

    def associate_annotations(self, entity_data, classified_annotations):
        if entity_data.get("Vertices") and len(entity_data["Vertices"]) >= 4:
            polygon = Polygon(entity_data["Vertices"])
            for key, annotations in classified_annotations.items():
                for annotation in annotations:
                    point = Point(annotation["Position"])
                    if polygon.contains(point):
                        if key not in entity_data:
                            entity_data[key] = []
                        entity_data[key].append(annotation["Text"])
        return entity_data

    def entities_to_dataframe(self, filename):
        data = []
        annotations = self.extract_annotations()
        classified_annotations = self.classify_annotations(annotations)
        for entity in self.msp:
            entity_data = {
                "Filename": filename,
                "Type": entity.dxftype(),
                "Layer": entity.dxf.layer,
                "Color": entity.dxf.color,
                "Vertices": [],
                "Text": None,
                "Position_X": None,
                "Position_Y": None,
                "Height": None,
                "Style": None,
                "Block": None,
                "Start_X": None,
                "Start_Y": None,
                "End_X": None,
                "End_Y": None,
                "Center_X": None,
                "Center_Y": None,
                "Radius": None,
                "Major_Axis_End_X": None,
                "Major_Axis_End_Y": None,
                "Minor_Axis_End_X": None,
                "Minor_Axis_End_Y": None
            }
            if entity.dxftype() == 'INSERT':
                block_name = entity.dxf.name
                block = self.doc.blocks.get(block_name)
                if block is None:
                    continue
                for block_entity in block:
                    block_entity_data = {
                        "Filename": filename,
                        "Type": block_entity.dxftype(),
                        "Layer": block_entity.dxf.layer,
                        "Color": block_entity.dxf.color,
                        "Block": block_name,
                        "Vertices": [],
                        "Text": None,
                        "Position_X": None,
                        "Position_Y": None,
                        "Height": None,
                        "Style": None,
                        "Start_X": None,
                        "Start_Y": None,
                        "End_X": None,
                        "End_Y": None,
                        "Center_X": None,
                        "Center_Y": None,
                        "Radius": None,
                        "Major_Axis_End_X": None,
                        "Major_Axis_End_Y": None,
                        "Minor_Axis_End_X": None,
                        "Minor_Axis_End_Y": None
                    }
                    self.update_entity_data(block_entity, block_entity_data)
                    block_entity_data = self.associate_annotations(block_entity_data, classified_annotations)
                    data.append(block_entity_data)
            else:
                self.update_entity_data(entity, entity_data)
                entity_data = self.associate_annotations(entity_data, classified_annotations)
                data.append(entity_data)
        return pd.DataFrame(data)

    def update_entity_data(self, entity, entity_data):
        if entity.dxftype() == "TEXT":
            entity_data.update({
                "Text": entity.dxf.text,
                "Position_X": entity.dxf.insert[0],
                "Position_Y": entity.dxf.insert[1],
                "Height": entity.dxf.height,
                "Style": entity.dxf.style
            })
        elif entity.dxftype() == "LINE":
            entity_data.update({
                "Start_X": entity.dxf.start[0],
                "Start_Y": entity.dxf.start[1],
                "End_X": entity.dxf.end[0],
                "End_Y": entity.dxf.end[1]
            })
        elif entity.dxftype() == "CIRCLE":
            entity_data.update({
                "Center_X": entity.dxf.center[0],
                "Center_Y": entity.dxf.center[1],
                "Radius": entity.dxf.radius
            })
        elif entity.dxftype() == "POLYLINE" or entity.dxftype() == "LWPOLYLINE":
            vertices = [(vertex.dxf.location.x, vertex.dxf.location.y) for vertex in entity.vertices] if entity.dxftype() == "POLYLINE" else [(point[0], point[1]) for point in entity.get_points()]
            entity_data.update({
                "Vertices": vertices
            })
        elif entity.dxftype() == "POINT":
            entity_data.update({
                "Position_X": entity.dxf.location[0],
                "Position_Y": entity.dxf.location[1]
            })
        elif entity.dxftype() == "ARC":
            entity_data.update({
                "Center_X": entity.dxf.center[0],
                "Center_Y": entity.dxf.center[1],
                "Radius": entity.dxf.radius,
                "Start_Angle": entity.dxf.start_angle,
                "End_Angle": entity.dxf.end_angle
            })
        elif entity.dxftype() == "ELLIPSE":
            entity_data.update({
                "Center_X": entity.dxf.center[0],
                "Center_Y": entity.dxf.center[1],
                "Major_Axis_End_X": entity.dxf.major_axis[0],
                "Major_Axis_End_Y": entity.dxf.major_axis[1],
                "Minor_Axis_End_X": entity.dxf.minor_axis[0],
                "Minor_Axis_End_Y": entity.dxf.minor_axis[1]
            })
        elif entity.dxftype() == "SPLINE":
            vertices = [(control_point[0], control_point[1]) for control_point in entity.control_points]
            entity_data.update({
                "Vertices": vertices
            })

    def print_all_entities(self):
        entity_types = set()
        for entity in self.msp:
            entity_types.add(entity.dxftype())
        print("All DXF entity types found:", entity_types)

if __name__ == "__main__":
    # Change pattern folder here
    #pattern = "basic_pattern"
    pattern = "" # If no pattern dir

    dxf_directory = "../data/02-07-2024-dxf-files/" + str(pattern) + "/"
    output_table_directory = "../data/output_tables/" + pattern + "_"

    dxf_loader = DXFLoader()

    all_data = []

    print(dxf_directory)

    for filename in os.listdir(dxf_directory):
        file_path = os.path.join(dxf_directory, filename)
        print(filename)
        if os.path.isfile(file_path) and filename.lower().endswith('.dxf'):
            dxf_loader.load_dxf(file_path)
            dxf_loader.print_all_entities()  # Print all entity types found in the current DXF file
            df = dxf_loader.entities_to_dataframe(filename)
            all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)
    sorted_df = combined_df.sort_values(by=['Filename', 'Type', 'Layer'])

    # Save the sorted DataFrame to CSV and Excel files
    sorted_df.to_csv(output_table_directory + 'combined_entities.csv', index=False)
    sorted_df.to_excel(output_table_directory + 'combined_entities.xlsx', index=False)

    print(sorted_df.head())  # Display the first few rows of the sorted DataFrame for verification
