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
tailoring-api/
├── app/         # Core application logic for file processing and alterations
│   ├── auto_grading.py          # Automated grading system
│   ├── create_table.py          # Preprocessing and table creation
│   └── piece_alteration_processor.py  # Pattern alterations
├── ui/          # User interface components for data ingestion
└── data/        # Local development data storage

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
bash
git clone https://github.com/your-org/tailoring-api.git
cd tailoring-api

### 2. Environment Setup
bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
pip install -r requirements.txt

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
