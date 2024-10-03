import pandas as pd
import matplotlib.pyplot as plt
import ast
import os
import seaborn as sns
import matplotlib.patheffects as pe
from utils.data_processing_utils import DataProcessingUtils
import numpy as np

class PlotGenerator:
    def __init__(self, output_dir="data/output/plots", dpi=300):
        self.output_dir = output_dir
        self.dpi = dpi
        self.data_processing_utils = DataProcessingUtils()

    def save_plot(self, fig, ax, plot_type, piece_name, alteration_rule=None):
        if plot_type == "vertices_plot":
            png_output_dir = os.path.join(self.output_dir, piece_name, "vertices", "png")
            svg_output_dir = os.path.join(self.output_dir, piece_name, "vertices", "svg")
        else:
            png_output_dir = os.path.join(self.output_dir, piece_name, alteration_rule, "png")
            svg_output_dir = os.path.join(self.output_dir, piece_name, alteration_rule, "svg")
        
        # Ensure both PNG and SVG directories exist
        os.makedirs(png_output_dir, exist_ok=True)
        os.makedirs(svg_output_dir, exist_ok=True)
        
        # Save as PNG
        png_output_file = os.path.join(png_output_dir, f"{plot_type}_{piece_name}.png")
        print(f"Saving {plot_type} as PNG to {png_output_file}")
        fig.savefig(png_output_file, format='png', dpi=self.dpi, bbox_inches='tight')
        print(f"PNG plot saved to {png_output_file}")

        # Get the current axis limits
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

        # Calculate the width and height in inches
        width_inches = x_max - x_min
        height_inches = y_max - y_min

        # Save as SVG
        svg_output_file = os.path.join(svg_output_dir, f"{plot_type}_{piece_name}.svg")
        print(f"Saving {plot_type} as SVG to {svg_output_file}")
        self.data_processing_utils.save_plot_as_svg(fig, ax, width_inches, height_inches, svg_output_file)
        print(f"SVG plot saved to {svg_output_file}")

        plt.close(fig)

    def plot_vertices(self, file_path_vertices, file_path):
        piece_name = os.path.splitext(os.path.basename(file_path_vertices))[0].split('_')[-1]
        alteration_rule = os.path.splitext(os.path.basename(file_path))[0].split('_')[-1]
        vertices_df = pd.read_csv(file_path_vertices)
        df = pd.read_csv(file_path)

        sns.set(style="whitegrid", context="notebook")
        fig, ax = plt.subplots(figsize=(50, 30))
        ax.set_facecolor('#f0f0f5')

        for _, row in vertices_df.iterrows():
            vertices = ast.literal_eval(row['vertices'])
            xs, ys = zip(*vertices)
            plt.plot(xs, ys, color='#0066ff', alpha=0.6, linewidth=3, linestyle='-', 
                     path_effects=[pe.withStroke(linewidth=5, foreground="white")])

        mtm_mask = df['mtm points'].notna()
        sns.scatterplot(x=df.loc[mtm_mask, 'pl_point_x'], y=df.loc[mtm_mask, 'pl_point_y'], 
                        color='orange', label='MTM Points', marker='D', s=250, alpha=0.7, 
                        edgecolor='black', ax=ax)

        for _, row in df[mtm_mask].iterrows():
            ax.text(row['pl_point_x'], row['pl_point_y'], str(int(row['mtm points'])),
                    fontsize=14, ha='center', va='center', color='black', weight='bold',
                    path_effects=[pe.withStroke(linewidth=3, foreground="white")])

        plt.title(f'Vertices Plot with MTM Points', 
                  fontsize=36, weight='bold', color='#4b4b4b', 
                  path_effects=[pe.withStroke(linewidth=3, foreground="white")])
        plt.xlabel('X Coordinate [in]', fontsize=24, weight='bold', color='#333333')
        plt.ylabel('Y Coordinate [in]', fontsize=24, weight='bold', color='#333333')
        plt.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.7)
        plt.tight_layout()
        plt.text(0.5, 0.97, f"Piece: {piece_name}", 
                 horizontalalignment='center', verticalalignment='center', 
                 transform=ax.transAxes, fontsize=28, weight='bold', color='#666666',
                 path_effects=[pe.withStroke(linewidth=2, foreground="white")])
        plt.tick_params(axis='both', which='major', labelsize=18)
        plt.legend(fontsize=22, loc='best', markerscale=2.5, shadow=True, frameon=True, fancybox=True)

        self.save_plot(fig, ax, "vertices_plot", piece_name)

    def plot_vertices_and_altered(self, file_path, file_path_vertices, 
                                  show_altered_points=True, show_notch_points=True, 
                                  show_altered_notch_points=True, show_mtm_points=True):
        piece_name = os.path.splitext(os.path.basename(file_path))[0].split('_')[0]
        alteration_rule = os.path.splitext(os.path.basename(file_path))[0].split('_')[-1]
        df = pd.read_csv(file_path)
        vertices_df = pd.read_csv(file_path_vertices)

        original_x = df['pl_point_x']
        original_y = df['pl_point_y']
        altered_x = df['pl_point_altered_x']
        altered_y = df['pl_point_altered_y']
        mtm_points = df['mtm points']

        mtm_mask = df['mtm points'].notna()

        if 'notch_labels' in df.columns and df['notch_labels'].dtype == object:
            notch_mask = df['notch_labels'].str.contains('notch', na=False)
            plot_notches = notch_mask.any()
        else:
            plot_notches = False

        sns.set(style="whitegrid", context="notebook")
        fig, ax = plt.subplots(figsize=(50, 30))
        ax.set_facecolor('#f0f0f5')

        if 'notch_labels' in df.columns and df['notch_labels'].dtype == object:
            notch_mask = df['notch_labels'].str.contains('notch', na=False)
            altered_notch_mask = notch_mask & (df['pl_point_altered_x'].notna() | df['pl_point_altered_y'].notna())
        else:
            notch_mask = pd.Series(False, index=df.index)
            altered_notch_mask = pd.Series(False, index=df.index)

        print(f"Number of altered notch points: {altered_notch_mask.sum()}")

        if show_altered_points:
            non_notch_altered_mask = (df['pl_point_altered_x'].notna() | df['pl_point_altered_y'].notna()) & ~altered_notch_mask
            sns.scatterplot(
                x=altered_x[non_notch_altered_mask], 
                y=altered_y[non_notch_altered_mask], 
                color='red', 
                label='Altered Points', 
                marker='x', 
                s=300,
                linewidth=4,
                alpha=0.8,
                edgecolor='black', 
                path_effects=[pe.withStroke(linewidth=5, foreground="white")]
            )

        if show_mtm_points:
            sns.scatterplot(x=original_x[mtm_mask], y=original_y[mtm_mask], color='orange', 
                            label='MTM Points', marker='D', s=250, alpha=0.7, edgecolor='black')

        sns.scatterplot(x=original_x, y=original_y, color='blue', label='Original Points', 
                        marker='o', s=150, alpha=0.5, edgecolor='black')

        if plot_notches and show_notch_points:
            sns.scatterplot(x=original_x[notch_mask], y=original_y[notch_mask], color='purple', 
                            label='Original Notch Points', marker='s', s=250, alpha=0.7, edgecolor='black')
        
        if plot_notches and show_altered_notch_points:
            sns.scatterplot(x=altered_x[altered_notch_mask], y=altered_y[altered_notch_mask], color='green', 
                            label='Altered Notch Points', marker='*', s=350, alpha=0.7, edgecolor='black')

        for _, row in vertices_df.iterrows():
            vertices = ast.literal_eval(row['vertices'])
            xs, ys = zip(*vertices)
            plt.plot(xs, ys, color='#0066ff', alpha=0.4, linewidth=2, linestyle='--')

        offset_x, offset_y = 0.15, 0.15

        if show_mtm_points:
            for i in range(len(original_x)):
                if pd.notna(mtm_points[i]):
                    plt.text(original_x[i] + offset_x, original_y[i] + offset_y, str(int(mtm_points[i])), 
                             fontsize=14, ha='center', color='black', weight='bold', 
                             path_effects=[pe.withStroke(linewidth=3, foreground="white")])

        if show_altered_points or show_altered_notch_points:
            for i in range(len(df)):
                if pd.notna(mtm_points[i]) and (pd.notna(altered_x[i]) or pd.notna(altered_y[i])):
                    x = altered_x[i] if pd.notna(altered_x[i]) else original_x[i]
                    y = altered_y[i] if pd.notna(altered_y[i]) else original_y[i]
                    
                    if altered_notch_mask[i] and show_altered_notch_points:
                        print(f"Labeling altered notch point: MTM {int(mtm_points[i])}, position ({x}, {y})")
                        plt.text(x + offset_x, y + offset_y, f"{int(mtm_points[i])} (AN)", 
                                 fontsize=14, ha='center', color='green', weight='bold',
                                 path_effects=[pe.withStroke(linewidth=3, foreground="white")])
                    elif show_altered_points and not altered_notch_mask[i]:
                        print(f"Labeling altered point: MTM {int(mtm_points[i])}, position ({x}, {y})")
                        plt.text(x + offset_x, y + offset_y, f"{int(mtm_points[i])} (A)", 
                                 fontsize=14, ha='center', color='red', weight='bold',
                                 path_effects=[pe.withStroke(linewidth=3, foreground="white")])

        plt.title(f'MTM, Altered, and Original Points with Vertices - {alteration_rule}', 
                  fontsize=28, weight='bold', color='#4b4b4b', 
                  path_effects=[pe.withStroke(linewidth=3, foreground="white")])
        plt.xlabel('X Coordinate [in]', fontsize=20, weight='bold', color='#333333')
        plt.ylabel('Y Coordinate [in]', fontsize=20, weight='bold', color='#333333')
        plt.legend(fontsize=22, loc='best', markerscale=2.5, shadow=True, frameon=True, fancybox=True, borderpad=1.5, labelspacing=1.2)
        plt.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.7)
        plt.tight_layout()

        self.save_plot(fig, ax, "vertices_and_altered", piece_name, alteration_rule)

    def generate_plots(self, file_path, file_path_vertices):
        self.plot_vertices_and_altered(file_path, file_path_vertices)
        self.plot_vertices(file_path_vertices, file_path)

if __name__ == "__main__":
    file_path = "data/staging_processed/debug/LGFG-SH-01-CCB-FO_1LTH-FULL.csv"
    #file_path = "data/staging_processed/debug/LGFG-SH-01-CCB-FO_4-WAIST.csv"
    #file_path = "data/staging_processed/debug/LGFG-SH-01-CCB-FO_7F-SHPOINT.csv"
    #file_path = "data/staging_processed/debug/LGFG-SH-01-CCB-FO_1LTH-FRONT.csv"

    file_path_vertices = "data/staging_processed/processed_vertices_by_piece/processed_vertices_LGFG-SH-01-CCB-FO.csv"
    
    plot_generator = PlotGenerator()
    plot_generator.generate_plots(file_path, file_path_vertices)