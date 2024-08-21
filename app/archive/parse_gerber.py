from gerber_renderer import Gerber

# Define the path to the Gerber files zip
file_path = "shirt-gerber-files.zip"

# Initialize the Gerber board object
board = Gerber.Board(file=file_path, max_height=500, verbose=True)

# Define the output path for the SVG files
output_path = '../data/output_svg/'

# Render the SVG files
board.render(output=output_path)

print(f"SVG files have been saved to {output_path}")
