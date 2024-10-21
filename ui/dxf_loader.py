import os
import ezdxf
import pandas as pd
from shapely.geometry import Point, Polygon
import io

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
            raise RuntimeError(f"Error loading DXF file '{self.file_path}': {e}")

    def extract_annotations(self):
        annotations = []
        for entity in self.msp:
            if entity.dxftype() == "TEXT":
                annotations.append({
                    "Text": entity.dxf.text,
                    "Point_X": entity.dxf.insert[0],
                    "Point_Y": entity.dxf.insert[1]
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
            position = (annotation["Point_X"], annotation["Point_Y"])
            if "Piece Name:" in text:
                classified_annotations["Piece_Name"].append({"Text": text, "Point_X": position[0], "Point_Y": position[1]})
            elif "Size:" in text:
                classified_annotations["Size"].append({"Text": text, "Point_X": position[0], "Point_Y": position[1]})
            elif "Annotation:" in text:
                classified_annotations["Annotation"].append({"Text": text, "Point_X": position[0], "Point_Y": position[1]})
            elif "Quantity:" in text:
                classified_annotations["Quantity"].append({"Text": text, "Point_X": position[0], "Point_Y": position[1]})
            elif "Category:" in text:
                classified_annotations["Category"].append({"Text": text, "Point_X": position[0], "Point_Y": position[1]})
        return classified_annotations

    def associate_annotations(self, entity_data, classified_annotations):
        if entity_data.get("Vertices") and len(entity_data["Vertices"]) >= 4:
            polygon = Polygon(entity_data["Vertices"])
            for key, annotations in classified_annotations.items():
                for annotation in annotations:
                    point = Point(annotation["Point_X"], annotation["Point_Y"])
                    if polygon.contains(point):
                        if key not in entity_data:
                            entity_data[key] = []
                        entity_data[key].append(annotation["Text"])
        return entity_data

    def entities_to_dataframe(self):
        data = []
        annotations = self.extract_annotations()
        classified_annotations = self.classify_annotations(annotations)
        point_label_counter = 0  # Initialize point label counter
        vertex_counter = 0  # Initialize vertex counter

        for entity in self.msp:
            if entity.dxftype() == 'INSERT':
                block_name = entity.dxf.name
                block = self.doc.blocks.get(block_name)
                if block is None:
                    continue
                for block_entity in block:
                    block_entity_data = {
                        "Filename": os.path.basename(self.file_path),
                        "Type": block_entity.dxftype(),
                        "Layer": block_entity.dxf.layer,
                        "Color": block_entity.dxf.color,
                        "Block": block_name,
                        "Text": None,
                        "Point_X": None,
                        "Point_Y": None,
                        "Height": None,
                        "Style": None,
                        "Line_Start_X": None,
                        "Line_Start_Y": None,
                        "Line_End_X": None,
                        "Line_End_Y": None,
                        "Vertices": None,  # Keep vertices as a list
                        "PL_POINT_X": None,
                        "PL_POINT_Y": None,
                        "Vertex_Index": None,
                        "Point Label": None,
                        "Vertex Label": None
                    }
                    self.update_entity_data(block_entity, block_entity_data)
                    block_entity_data = self.associate_annotations(block_entity_data, classified_annotations)
                    if block_entity.dxftype() in ["POLYLINE", "LWPOLYLINE"]:
                        vertex_label = vertex_counter  # Assign a unique ID for each vertex
                        for idx, (x, y) in enumerate(block_entity_data["Vertices"]):
                            row_data = block_entity_data.copy()
                            row_data.update({
                                "PL_POINT_X": x,
                                "PL_POINT_Y": y,
                                "Vertex_Index": idx,
                                "Point Label": point_label_counter,  # Use point label counter
                                "Vertex Label": vertex_label  # Use vertex counter
                            })
                            point_label_counter += 1  # Increment point label counter
                            data.append(row_data)
                        vertex_counter += 1  # Increment vertex counter for next unique vertex
                    else:
                        block_entity_data.update({"Point Label": point_label_counter})
                        point_label_counter += 1
                        data.append(block_entity_data)
            else:
                entity_data = {
                    "Filename": os.path.basename(self.file_path),
                    "Type": entity.dxftype(),
                    "Layer": entity.dxf.layer,
                    "Color": entity.dxf.color,
                    "Text": None,
                    "Point_X": None,
                    "Point_Y": None,
                    "Height": None,
                    "Style": None,
                    "Block": None,  # Initialize as None instead of "Unnamed_Block"
                    "Line_Start_X": None,
                    "Line_Start_Y": None,
                    "Line_End_X": None,
                    "Line_End_Y": None,
                    "Vertices": None,  # Keep vertices as a list
                    "PL_POINT_X": None,
                    "PL_POINT_Y": None,
                    "Vertex_Index": None,
                    "Point Label": None,
                    "Vertex Label": None
                }
                self.update_entity_data(entity, entity_data)
                entity_data = self.associate_annotations(entity_data, classified_annotations)
                if entity.dxftype() in ["POLYLINE", "LWPOLYLINE"]:
                    vertex_label = vertex_counter  # Assign a unique ID for each vertex
                    for idx, (x, y) in enumerate(entity_data["Vertices"]):
                        row_data = entity_data.copy()
                        row_data.update({
                            "PL_POINT_X": x,
                            "PL_POINT_Y": y,
                            "Vertex_Index": idx,
                            "Point Label": point_label_counter,  # Use point label counter
                            "Vertex Label": vertex_label  # Use vertex counter
                        })
                        point_label_counter += 1  # Increment point label counter
                        data.append(row_data)
                    vertex_counter += 1  # Increment vertex counter for next unique vertex
                elif entity.dxftype() == "LINE":
                    entity_data.update({
                        "Line_Start_X": entity.dxf.start[0],
                        "Line_Start_Y": entity.dxf.start[1],
                        "Line_End_X": entity.dxf.end[0],
                        "Line_End_Y": entity.dxf.end[1],
                        "Point Label": point_label_counter
                    })
                    point_label_counter += 1
                    data.append(entity_data)
                else:
                    entity_data.update({"Point Label": point_label_counter})
                    point_label_counter += 1
                    data.append(entity_data)
        
        df = pd.DataFrame(data)
        
        # Drop unnecessary columns
        columns_to_drop = ['Center_X', 'Center_Y', 'Radius', 'Major_Axis_End_X', 'Major_Axis_End_Y', 'Minor_Axis_End_X', 'Minor_Axis_End_Y']
        df = df.drop(columns=columns_to_drop, errors='ignore')

        # Add an empty column for MTM Points
        df['MTM Points'] = ""

        return df

    def update_entity_data(self, entity, entity_data):
        if entity.dxftype() == "TEXT":
            entity_data.update({
                "Text": entity.dxf.text,
                "Point_X": entity.dxf.insert[0],
                "Point_Y": entity.dxf.insert[1],
                "Height": entity.dxf.height,
                "Style": entity.dxf.style
            })
        elif entity.dxftype() == "LINE":
            entity_data.update({
                "Line_Start_X": entity.dxf.start[0],
                "Line_Start_Y": entity.dxf.start[1],
                "Line_End_X": entity.dxf.end[0],
                "Line_End_Y": entity.dxf.end[1]
            })
        elif entity.dxftype() == "CIRCLE":
            entity_data.update({
                "Point_X": entity.dxf.center[0],
                "Point_Y": entity.dxf.center[1],
                "Radius": entity.dxf.radius
            })
        elif entity.dxftype() == "POLYLINE" or entity.dxftype() == "LWPOLYLINE":
            vertices = [(vertex.dxf.location.x, vertex.dxf.location.y) for vertex in entity.vertices] if entity.dxftype() == "POLYLINE" else [(point[0], point[1]) for point in entity.get_points()]
            entity_data.update({
                "Vertices": vertices
            })
        elif entity.dxftype() == "POINT":
            entity_data.update({
                "Point_X": entity.dxf.location[0],
                "Point_Y": entity.dxf.location[1]
            })
        elif entity.dxftype() == "ARC":
            entity_data.update({
                "Point_X": entity.dxf.center[0],
                "Point_Y": entity.dxf.center[1],
                "Radius": entity.dxf.radius,
                "Start_Angle": entity.dxf.start_angle,
                "End_Angle": entity.dxf.end_angle
            })
        elif entity.dxftype() == "ELLIPSE":
            entity_data.update({
                "Point_X": entity.dxf.center[0],
                "Point_Y": entity.dxf.center[1],
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

    dxf_filename = "LGFG-SH-01-CCB-FOA-GRADED.dxf"
    dxf_filepath = "data/input/" + dxf_filename
    output_table_directory = "data/output/processed_dxf/" 

    dxf_loader = DXFLoader()
    dxf_loader.load_dxf(dxf_filepath)
    df = dxf_loader.entities_to_dataframe()
    sorted_df = df.sort_values(by=['Filename', 'Type', 'Layer'])

    # Save the sorted DataFrame to CSV and Excel files
    sorted_df.to_csv(output_table_directory + 'combined_entities.csv', index=False)
    sorted_df.to_excel(output_table_directory + 'combined_entities.xlsx', index=False)

    print(sorted_df.head())  # Display the first few rows of the sorted DataFrame for verification
