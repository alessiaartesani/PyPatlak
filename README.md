# PyPatlak: A Command-Line Tool for Patlak Graphical Analysis

This tool provides a simple and robust command-line interface for performing voxel-wise Patlak analysis on dynamic PET data. It is designed to be easy to install and use, making quantitative PET analysis more accessible for researchers.

## Key Features

-   **DICOM to NIfTI Conversion**: Automatically handles the conversion of dynamic PET DICOM series to a single 4D NIfTI file using the dcm2niix tool.
-   **Patient-Specific Input Function**: Scales a population-based input function (PBIF) with a patient-specific Arterial Input Function (AIF) mean extracted from a provided mask.
-   **Voxel-wise Patlak Plot**: Computes Ki (uptake rate constant) and Vd (distribution volume) maps by performing a linear regression on the Patlak plot for each voxel.
-   **NIfTI Output**: Saves the resulting Ki and Vd maps as standard NIfTI files, compatible with most medical imaging software.

## Installation 💻

To get started, you'll need **Python 3.8+** and the **dcm2niix** executable installed on your system.

Create a virtual environment and activate it (recommended):

**Windows:**
```bash
python -m venv mypatlak_env
mypatlak_env\Scripts\activate
```
**Linux/Mac:**
```bash
python -m venv mypatlak_env
source mypatlak_env/bin/activate  
```

Simply use ```pip``` to install the package from PyPI:

```bash
pip install .
```
  
## Usage 🚀

The `pypatlak` command-line tool is designed for ease of use.

```bash
pypatlak <dicom_dir> <aif_mask_path> <output_dir> <patient_id> [options]
```

**Arguments**

```<dicom_dir>```: Path to the directory containing the dynamic PET DICOM files.

```<aif_mask_path>```: Path to the NIfTI file (.nii or .nii.gz) for the AIF mask.

```<output_dir>```: Path to the directory where the output maps will be saved.

```<patient_id>```: A unique identifier for the patient, used for naming output files.

**Options**

```--rf``` RECOVERY_FACTOR: Correct for the partial volume effect and scale AIF mask mean values (default: 1.0)

```--input-function``` INPUT_FUNCTION: Path to the population-based input function CSV file. (default: ./input_function.csv)

```--dcm2niix-path``` DCM2NIIX_PATH: Path to the dcm2niix executable. (default: dcm2niix, assumes it's in your system's PATH)

**Example**

To run the analysis on a dataset:
```bash
pypatlak /path/to/my/dicoms /path/to/my/aif_mask.nii.gz /path/to/my/dicoms/results my_patient_01 --rf 1.2
```

## Required Folder Structure 📂

To run the analysis, your input data should be organized as follows:

```bash
.
├── patient_data/
│   ├── patient_1/
│   │   ├── pt01_DICOM/  # Directory containing DICOM files
│   │   └── pt01_aif.nii.gz # AIF mask file
│   └── patient_2/
│       ├── pt02_DICOM/
│       └── pt02_aif.nii.gz
├── input_function.csv   # The population-based input function
└── ...
```

## Resultant Folder Structure 📂

After the analysis is complete, the specified output directory will contain the generated parametric maps:

```bash
results/
├── Ki_map.nii
└── Vd_map.nii
```

## Citation 📝

If you use this software in your research, please cite the following repository and/or paper:

Repository Citation:

**BibTeX:**
```bash
@software{AArtesani_PyPatlak_2025,
  author = {Alessia Artesani},
  title = {{PyPatlak: A Command-Line Tool For Patlak Analysis}},
  url = {https://github.com/alessiaartesani/pypatlak},
  year = {2025},
}
```

**Paper Citation:**

pyPatlak: Open-Source Voxel-Wise Patlak Analysis of Dynamic PET Data
Alessia Artesani
medRxiv 2025.09.16.25335861; doi: https://doi.org/10.1101/2025.09.16.25335861

**License**

This project is licensed under the MIT License. 

**Acknowledgements**

This tool uses the nibabel, pydicom, numpy, and scipy libraries.
The DICOM to NIfTI conversion is powered by dcm2niix.

**Contributors**

Alessia Artesani - Initial development and project setup.
