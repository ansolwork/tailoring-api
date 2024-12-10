# Tailoring-api
By The AI Framework. 

## Introduction
The Tailoring-api is a specialized system designed to generate alterations for LGFG, with a primary focus on full shirt pieces. The system handles data ingestion through a UI, preprocesses pattern data, and generates altered shirt pieces that can be exported as Plotter files (PLT/HPGL) for factory production.

## Key Features
- DXF file parsing and processing
- Automated pattern grading system
- MTM (Made-to-Measure) point handling
- AWS integration for file storage
- UI for data ingestion
- PLT/HPGL output file generation

## Project Structure
```
tailoring-api/
├── app/                                # Core application logic
│   ├── auto_grading.py                # Automated grading system
│   ├── create_table.py                # Preprocessing and table creation
│   ├── piece_alteration_processor.py  # Pattern alterations
│   ├── aws_load.py                    # AWS S3 file loading utility
│   ├── aws_save.py                    # AWS S3 file saving utility
│   ├── api.py                         # REST API for processing alterations
│   └── plot_generator.py              # Visualization tool for pattern pieces
├── ui/                                # User interface components
└── data/                              # Local development data
    ├── debug/                         # Debugging output and intermediate files
    ├── input/                         # Raw input files
    │   ├── mtm_points.xlsx           # MTM point specifications
    │   ├── mtm_combined_entities_labeled/  # Base size labeled entities
    │   └── graded_mtm_combined_entities_labeled/  # Graded size labeled entities
    ├── output/                        # Final processed outputs
    │   └── plots/                     # Generated visualization plots
    ├── staging/                       # Intermediate processing data
    │   ├── base/                      # Base size processing
    │   │   └── shirt/                 # Shirt-specific data
    │   │       ├── alteration_by_piece/
    │   │       ├── combined_alteration_tables/
    │   │       └── vertices/
    │   └── graded/                    # Graded sizes processing
    │       └── shirt/                 # Shirt-specific data
    │           ├── alteration_by_piece/
    │           ├── combined_alteration_tables/
    │           └── vertices/
    └── staging_processed/             # Final stage processing
        ├── base/                      # Base size final processing
        └── graded/                    # Graded sizes final processing
```

## Data Directory Structure

### /data/input/
- Contains raw input files and labeled entities
- Includes MTM point specifications and combined entities
- Separate directories for base and graded sizes

### /data/staging/
- Intermediate processing stage
- Organized by base and graded sizes
- Contains:
  - Alteration tables by piece
  - Combined alteration tables
  - Vertices information

### /data/staging_processed/
- Final processing stage
- Contains processed alterations and vertices
- Organized by base and graded sizes
- Includes debug information for alterations

### /data/output/
- Final output files
- Contains generated plots and visualizations
- Organized by piece and alteration type

### /data/debug/
- Debugging information and intermediate files
- Helpful for troubleshooting and development
- Contains step-by-step processing information

## Input/Output Specifications

### Input
- DXF files containing pattern pieces
- Excel files with MTM points and alteration rules
- Configuration files (YAML):
  - `tailoring_api_config.yaml`: Main configuration
  - `config_mtm.yaml`: MTM reference configuration
  - `config_graded.yaml`: Grading configuration

### Output
- Processed pattern pieces
- PLT/HPGL files for factory production
- CSV files containing:
  - Alteration rules
  - Pattern piece data
  - Vertices information

## Dependencies
- python>=3.8
- boto3
- pandas
- numpy
- psycopg2
- pyyaml
- ezdxf
- matplotlib
- networkx

## Prerequisites
- Python 3.8+
- AWS Account
- PostgreSQL
- Docker (optional)

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-org/tailoring-api.git
cd tailoring-api
```

### 2. Environment Setup
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. AWS Setup
1. Create an AWS account
2. Create an S3 bucket for file storage
3. Configure AWS credentials:
```bash
aws configure
# Enter your AWS Access Key ID
# Enter your AWS Secret Access Key
# Enter your default region
# Enter your output format
```

### 4. Data Processing Pipeline
1. Load reference and graded files:
```bash
python -m app.aws_load config_mtm.yml
python -m app.aws_load config_graded_mtm.yml
```

2. Create preprocessing tables:
```bash
python -m app.create_table
```

3. Process piece alterations:
```bash
python -m piece_alteration_processor
```

## Docker Support
We provide two Dockerfiles for different components:

### UI Container
```bash
# Build UI container
docker build -f Dockerfile.ui -t tailoring-ui .
```

### App Container
```bash
# Build app container
docker build -f Dockerfile.app -t tailoring-app .
```

### Running with Docker Compose
```bash
docker-compose up
```

## Development Workflow
1. Upload DXF files through the UI
2. Process files to generate unlabeled output
3. Manually label MTM points (8000, 8001, etc.)
4. Re-upload labeled files
5. Run alteration code to generate modified patterns

## Terminology
- **DXF**: Design Exchange Format, used for CAD pattern files
- **MTM Points**: Made-to-Measure points (8000, 8001, etc.) used for alterations
- **PLT/HPGL**: Plotter file formats used for factory production
- **Base Size**: The reference size from which other sizes are graded
- **Grading**: The process of scaling patterns to different sizes
- **Notch Points**: Reference points used in pattern construction

## Configuration Files
### tailoring_api_config.yaml
- AWS S3 bucket configuration
- File path settings
- Database credentials
- Allowed file types

### config_mtm.yaml
- MTM point definitions
- Reference measurements
- Alteration rules

### config_graded.yaml
- Grading rules
- Size specifications
- Scale factors

## API Documentation
[Coming soon]

## Contributing
[Coming soon]

## License
[Specify license]

## Support
For support, please contact [contact information]

## Code Description

### auto_grading.py
Automated grading system that:
- Takes base pattern size (e.g., size 39)
- Generates patterns for other sizes
- Applies predefined grading rules
- Processes sizes sequentially

### create_table.py
Preprocessing system that:
- Creates structured database of pattern pieces
- Processes MTM points and alteration rules
- Generates CSV files for:
  - Alteration rules
  - Pattern pieces
  - Vertices data

### piece_alteration_processor.py
Core alteration engine that:
- Processes alterations on pattern pieces
- Handles MTM point movements
- Maintains pattern integrity
- Generates output files

### aws_load.py
AWS S3 loading utility that:
- Loads files from S3 bucket using boto3
- Handles Excel and CSV file formats
- Implements parallel file loading with ThreadPoolExecutor
- Includes file size checks and optimizations
- Saves downloaded files to local directory

### aws_save.py
AWS S3 saving utility that:
- Uploads local files to S3 bucket
- Maintains directory structure when uploading
- Supports Excel and CSV file formats
- Handles error logging and reporting
- Configurable through YAML configuration files

### api.py
REST API for processing alterations that:
- Provides endpoints to apply alterations and retrieve entities
- Uses Flask for API routing
- Connects to a PostgreSQL database using SQLAlchemy
- Handles JSON and YAML input formats
- Computes alteration amounts based on input measurements
- Logs detailed information for debugging and monitoring

### plot_generator.py
Visualization tool that:
- Generates detailed plots of pattern pieces
- Features:
  - Plots original vertices and MTM points
  - Shows altered points and notch points
  - Visualizes alterations with before/after comparisons
  - Supports both PNG and SVG output formats
- Key functionalities:
  - `plot_vertices`: Displays original pattern with MTM points
  - `plot_vertices_and_altered`: Shows comparison of original and altered patterns
  - Includes detailed labeling and color-coding for different point types
  - Saves high-resolution outputs with configurable DPI

## Configuration Examples

### tailoring_api_config.yaml
```yaml
aws:
  bucket: your-bucket-name
  region: your-region
database:
  host: localhost
  port: 5432
  name: db_name
```

### config_mtm.yaml
```yaml
mtm_points:
  - id: 8000
    description: "Shoulder point"
  - id: 8001
    description: "Chest point"
```

### config_graded.yaml
```yaml
sizes:
  - 38
  - 39
  - 40
grading_rules:
  chest: 1.0
  waist: 0.8
```

## Troubleshooting

### Common Issues

#### 1. AWS Configuration
- Error: "Unable to locate credentials"
- Solution: Run `aws configure` and enter credentials

#### 2. Database Connection
- Error: "Could not connect to database"
- Solution: Check PostgreSQL service is running

#### 3. Docker Issues
- Error: "Container failed to start"
- Solution: Check Docker daemon is running
