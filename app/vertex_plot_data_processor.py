import ast
import hashlib
import pandas as pd
import os

class VertexPlotDataProcessor:
    """
    A class responsible for processing vertex and alteration data, scaling vertices, 
    and preparing plot data for visualization.
    """
    def __init__(self, data_processing_utils, scaling_factor=1):
        """
        Initialize with the data processing utilities and a scaling factor.

        Parameters:
        - data_processing_utils: Utility class instance for data operations.
        - scaling_factor: Factor by which to scale vertices (default is 1).
        """
        self.data_processing_utils = data_processing_utils
        self.scaling_factor = scaling_factor
        self.scaled_unique_vertices = []
        self.scaled_altered_vertices = []
        self.scaled_altered_vertices_reduced = []
        self.plot_df = ""

    # --------------------------- Plot Data Preparation --------------------------- #
    def initialize_plot_data(self):
        """
        Initializes an empty dictionary to hold plot data.

        This method:
        - Prepares a structure to store information such as unique vertices, altered vertices, 
          and MTM (machine tool movement) points.
        - The dictionary is used to accumulate and organize data before plotting.

        Returns:
        --------
        plot_data : dict
            A dictionary with empty lists for various plot data types:
            - 'unique_vertices', 'unique_vertices_xs', 'unique_vertices_ys'
            - 'altered_vertices', 'altered_vertices_xs', 'altered_vertices_ys'
            - 'altered_vertices_reduced', 'altered_vertices_reduced_xs', 'altered_vertices_reduced_ys'
            - 'mtm_points'
        """
        return {
            'unique_vertices': [],
            'unique_vertices_xs': [],
            'unique_vertices_ys': [],
            'altered_vertices': [],
            'altered_vertices_xs': [],
            'altered_vertices_ys': [],
            'altered_vertices_reduced': [],
            'altered_vertices_reduced_xs': [],
            'altered_vertices_reduced_ys': [],
            'mtm_points': []
        }

    def prepare_plot_data(self, df, vertices_df, output_dir="data/staging_processed/"):
        """
        Processes the vertices and MTM points to prepare the data for plotting.

        This method:
        - Iterates over the main data (`df`) and vertices data (`vertices_df`).
        - Scales, transforms, and stores the vertices and MTM points in a plot-ready format.
        - Saves the processed plot data to an Excel file in the specified output directory.

        Parameters:
        ----------
        df : pandas.DataFrame
            The DataFrame containing the main data (e.g., alterations).
        
        vertices_df : pandas.DataFrame
            The DataFrame containing the vertices data.

        output_dir : str, optional
            The directory path to save the processed plot data (default is "data/staging_processed/").
        """
        plot_data = self.initialize_plot_data()

        df = df.copy()
        vertices_df = vertices_df.copy()
        
        # Process main data
        #for _, row in df.iterrows():
        #    try:
                #self.process_altered_vertices(row, plot_data, self.scaled_altered_vertices)
                #self.process_altered_vertices_reduced(row, plot_data, self.scaled_altered_vertices_reduced)
                #self.process_mtm_points(row, plot_data)
        #    except Exception as e:
        #        continue
        
        # Process vertices data
        for _, row in vertices_df.iterrows():
            try:
                self.process_vertices(row, plot_data, self.scaled_unique_vertices)
            except Exception as e:
                continue

        plot_data['unique_vertices'] = self.scaled_unique_vertices
        
        # Save the prepared plot data
        self.save_plot_data(plot_data, output_dir)

    # --------------------------- Data Processing Methods --------------------------- #
    def scale_vertices(self, vertices_list):
        """
        Scales a list of (x, y) coordinates by the specified scaling factor.

        This method:
        - Separates the x and y coordinates from the input list of vertex tuples.
        - Applies the scaling factor to both x and y coordinates.

        Parameters:
        ----------
        vertices_list : list of tuple
            A list of tuples representing (x, y) coordinates of vertices.

        Returns:
        --------
        tuple of lists
            A tuple containing two lists: the scaled x coordinates and the scaled y coordinates.
        """
        xs, ys = zip(*vertices_list)
        return self.data_processing_utils.scale_coordinates(xs, ys, self.scaling_factor)

    def process_vertices(self, row, plot_data, scaled_vertices):
        """
        Processes and scales original vertices, adding them to the plot data.

        This method:
        - Extracts original vertices from the input row.
        - Scales the vertices based on the scaling factor.
        - Adds the scaled vertices to the plot data if they are unique (based on a hash).

        Parameters:
        ----------
        row : pandas.Series
            A row of data containing original vertices.

        plot_data : dict
            The dictionary used to store the processed plot data.

        scaled_vertices : list
            A list to store the scaled vertices.
        """
        vertices_list = ast.literal_eval(row['vertices'])
        
        if vertices_list:
            xs, ys = self.scale_vertices(vertices_list)

            # Create a unique identifier for the vertex list
            scaled_vertex_tuple = tuple(zip(xs, ys))
            vertex_hash = hashlib.md5(str(scaled_vertex_tuple).encode()).hexdigest()

            if vertex_hash not in plot_data['unique_vertices']:
                plot_data['unique_vertices'].append(vertex_hash)
                plot_data['unique_vertices_xs'].append(xs)
                plot_data['unique_vertices_ys'].append(ys)
                scaled_vertices.append(scaled_vertex_tuple)

    def process_altered_vertices(self, row, plot_data, scaled_vertices):
        """
        Processes and scales altered vertices, adding them to the plot data.

        This method:
        - Extracts the altered vertices from the row.
        - Scales the vertices and adds them to the plot data if they haven't been processed already.

        Parameters:
        ----------
        row : pandas.Series
            A row of data containing altered vertices.

        plot_data : dict
            The dictionary used to store the processed plot data.

        scaled_vertices : list
            A list to store the scaled altered vertices.
        """
        raw_altered = row['altered_vertices']
        if pd.notna(raw_altered):
            altered_vertices_list = ast.literal_eval(raw_altered)
            if altered_vertices_list and altered_vertices_list not in plot_data['altered_vertices']:
                plot_data['altered_vertices'].append(altered_vertices_list)
                xs, ys = zip(*altered_vertices_list)
                xs, ys = self.data_processing_utils.scale_coordinates(xs, ys, self.scaling_factor)
                plot_data['altered_vertices_xs'].append(xs)
                plot_data['altered_vertices_ys'].append(ys)
                scaled_vertices.append(tuple(zip(xs, ys)))

    def process_altered_vertices_reduced(self, row, plot_data, scaled_vertices):
        """
        Processes and scales reduced altered vertices, adding them to the plot data.

        This method:
        - Extracts the reduced altered vertices from the row.
        - Scales the reduced vertices and adds them to the plot data.

        Parameters:
        ----------
        row : pandas.Series
            A row of data containing reduced altered vertices.

        plot_data : dict
            The dictionary used to store the processed plot data.

        scaled_vertices : list
            A list to store the scaled reduced altered vertices.
        """
        raw_altered_reduced = row['altered_vertices_reduced']
        if not pd.isna(raw_altered_reduced):
            altered_vertices_list_reduced = ast.literal_eval(raw_altered_reduced)
            if altered_vertices_list_reduced and altered_vertices_list_reduced not in plot_data['altered_vertices_reduced']:
                plot_data['altered_vertices_reduced'].append(altered_vertices_list_reduced)
                xs, ys = zip(*altered_vertices_list_reduced)
                xs, ys = self.data_processing_utils.scale_coordinates(xs, ys, self.scaling_factor)
                plot_data['altered_vertices_reduced_xs'].append(xs)
                plot_data['altered_vertices_reduced_ys'].append(ys)
                scaled_vertices.append(tuple(zip(xs, ys)))

    def process_mtm_points(self, row, plot_data):
        """
        Processes and scales the MTM (Machine Tool Movement) points, adding them to the plot data.

        This method:
        - Extracts and scales the MTM points from the row.
        - Adds both the original and altered MTM points to the plot data.

        Parameters:
        ----------
        row : pandas.Series
            A row of data containing MTM points and relevant attributes.
        
        plot_data : dict
            The dictionary used to store the processed plot data.
        """
        def add_point(x, y, x_old, y_old, label, mtm_dependent, mtm_dependent_x, mtm_dependent_y, 
                    color, movement_x=0., movement_y=0., nearby_point_left=None, nearby_point_right=None,
                    nearby_point_left_coords=None, nearby_point_right_coords=None):
            if x is not None and y is not None:
                plot_data['mtm_points'].append({
                    'x': x * self.scaling_factor,
                    'y': y * self.scaling_factor,
                    'x_old': x_old * self.scaling_factor,
                    'y_old': y_old * self.scaling_factor,
                    'movement_x': movement_x,
                    'movement_y': movement_y,
                    'mtm_dependent': mtm_dependent,
                    'mtm_dependent_x': mtm_dependent_x,
                    'mtm_dependent_y': mtm_dependent_y,
                    'label': str(int(label)),
                    'belongs_to': 'original' if color == 'red' else 'altered',
                    'nearby_point_left': nearby_point_left,
                    'nearby_point_right': nearby_point_right,
                    'nearby_point_left_coords': nearby_point_left_coords,
                    'nearby_point_right_coords': nearby_point_right_coords,
                    'color': color
                })

        mtm_points = row.get('mtm points')
        if not pd.isna(mtm_points):
            # Add the original point
            add_point(x=row['pl_point_x'], y=row['pl_point_y'], 
                    x_old=row['pl_point_x'], y_old=row['pl_point_y'], 
                    label=row['mtm points'], mtm_dependent=row['mtm_dependant'], 
                    mtm_dependent_x=row['mtm_dependant_x'], mtm_dependent_y=row['mtm_dependant_y'], 
                    color='red')

            # Add the altered point, if applicable
            mtm_point_alteration = row.get('mtm_points_alteration')
            if pd.notna(mtm_point_alteration):
                mtm_new_coords = ast.literal_eval(row['new_coordinates'])

                # Safely handle inbetween_points to avoid index errors
                inbetween_points = ast.literal_eval(row.get('mtm_points_in_altered_vertices', '[]'))
                if len(inbetween_points) >= 3:
                    add_point(x=mtm_new_coords[0], y=mtm_new_coords[1], 
                            x_old=row['pl_point_x'], y_old=row['pl_point_y'],
                            label=mtm_point_alteration, mtm_dependent=row['mtm_dependant'], 
                            mtm_dependent_x=row['mtm_dependant_x'], mtm_dependent_y=row['mtm_dependant_y'],
                            movement_x=row['movement_x'], movement_y=row['movement_y'],
                            nearby_point_left=inbetween_points[0]['mtm_point'], nearby_point_right=inbetween_points[2]['mtm_point'],
                            nearby_point_left_coords=inbetween_points[0]['original_coordinates'], 
                            nearby_point_right_coords=inbetween_points[2]['original_coordinates'],
                            color='blue')
                else:
                    # Handle the case where inbetween_points is not as expected
                    add_point(x=mtm_new_coords[0], y=mtm_new_coords[1], 
                            x_old=row['pl_point_x'], y_old=row['pl_point_y'],
                            label=mtm_point_alteration, mtm_dependent=row['mtm_dependant'], 
                            mtm_dependent_x=row['mtm_dependant_x'], mtm_dependent_y=row['mtm_dependant_y'],
                            color='blue', movement_x=row['movement_x'], movement_y=row['movement_y'])

        # Process mtm_points_in_altered_vertices
        mtm_points_in_altered_vertices = ast.literal_eval(row.get('mtm_points_in_altered_vertices', '[]'))
        for mtm_info in mtm_points_in_altered_vertices:
            original_coords = mtm_info.get('original_coordinates')
            altered_coords = mtm_info.get('altered_coordinates')
            mtm_label = mtm_info.get('mtm_point')

            # Add the original coordinates if present
            if original_coords and original_coords[0] is not None and original_coords[1] is not None:
                add_point(x=original_coords[0], y=original_coords[1], 
                        x_old=original_coords[0], y_old=original_coords[1],
                        label=mtm_label, mtm_dependent=row['mtm_dependant'], 
                        mtm_dependent_x=row['mtm_dependant_x'], mtm_dependent_y=row['mtm_dependant_y'], 
                        color='red')
                    
            if altered_coords and altered_coords[0] is not None and altered_coords[1] is not None:
                # Safely handle inbetween_points again
                if len(inbetween_points) >= 3:
                    add_point(altered_coords[0], altered_coords[1], 
                            original_coords[0], original_coords[1], 
                            mtm_label, row['mtm_dependant'], mtm_dependent_x=row['mtm_dependant_x'], mtm_dependent_y=row['mtm_dependant_y'], 
                            movement_x=row['movement_x'], movement_y=row['movement_y'],
                            nearby_point_left=inbetween_points[0]['mtm_point'], nearby_point_right=inbetween_points[2]['mtm_point'],
                            nearby_point_left_coords=inbetween_points[0]['original_coordinates'], 
                            nearby_point_right_coords=inbetween_points[2]['original_coordinates'],
                            color='blue')
                else:
                    add_point(altered_coords[0], altered_coords[1], 
                            original_coords[0], original_coords[1], 
                            mtm_label, row['mtm_dependant'], mtm_dependent_x=row['mtm_dependant_x'], mtm_dependent_y=row['mtm_dependant_y'], 
                            color='blue', movement_x=row['movement_x'], movement_y=row['movement_y'])

    # --------------------------- Data Saving Methods --------------------------- #
    def save_plot_data(self, plot_data, output_dir):
        """
        Saves the processed plot data into an Excel file.

        This method:
        - Converts the plot data into DataFrames for unique vertices, altered vertices, and MTM points.
        - Saves the resulting DataFrame into an Excel file in the specified output directory.

        Parameters:
        ----------
        plot_data : dict
            The dictionary containing the processed plot data.

        output_dir : str
            The directory where the plot data should be saved as an Excel file.
        """
        df_unique = pd.DataFrame({
            'unique_vertices': plot_data['unique_vertices'],
            'unique_vertices_x': plot_data['unique_vertices_xs'],
            'unique_vertices_y': plot_data['unique_vertices_ys']
        })

        df_altered = pd.DataFrame({
            'altered_vertices': plot_data['altered_vertices'],
            'altered_vertices_x': plot_data['altered_vertices_xs'],
            'altered_vertices_y': plot_data['altered_vertices_ys'],
            'altered_vertices_reduced': plot_data['altered_vertices_reduced'],
            'altered_vertices_reduced_x': plot_data['altered_vertices_reduced_xs'],
            'altered_vertices_reduced_y': plot_data['altered_vertices_reduced_ys']
        })

        df_mtm_points = pd.DataFrame({
            'mtm_points': plot_data['mtm_points']
        })

        plot_df = pd.concat([df_unique, df_altered, df_mtm_points], axis=1)
        plot_df.columns = ['unique_vertices', 'unique_vertices_x', 'unique_vertices_y', 
                           'altered_vertices', 'altered_vertices_x', 'altered_vertices_y',
                           'altered_vertices_reduced', 'altered_vertices_reduced_x', 'altered_vertices_reduced_y',
                           'mtm_points']
        
        plot_df = plot_df.drop_duplicates(subset=["mtm_points"])

        output_path = os.path.join(output_dir, "unique_vertices.xlsx")
        plot_df.to_excel(output_path, index=False)
        print(f"Unique vertices saved to {output_path}")
        self.plot_df = plot_df
