import os
import subprocess
from pathlib import Path
import numpy as np
import nibabel as nib
from nibabel.orientations import aff2axcodes
import pydicom
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from datetime import datetime
import shutil

# --- DICOM UTILITY FUNCTIONS ---

def extract_dicom_info(dicom_file: Path) -> tuple:
    """
    Extracts key information from a DICOM file header for dynamic PET analysis.

    Args:
        dicom_file (Path): Path to the DICOM file.

    Returns:
        tuple: A tuple containing (delay_inj_acq, injected_dose, patient_weight, nFrames, nSlices).
    """
    ds = pydicom.dcmread(dicom_file, stop_before_pixels=True)
    try:
        rseq = ds.RadiopharmaceuticalInformationSequence[0]
        injected_dose = float(rseq.RadionuclideTotalDose)  # in Bq
        injection_time = rseq.RadiopharmaceuticalStartTime
        acquisition_time = ds.AcquisitionTime
        patient_weight = float(ds.PatientWeight)  # in kg
        nFrames = ds.NumberOfTimeSlices
        nSlices = ds.NumberOfSlices
        hl = rseq.RadionuclideHalfLife  # Half-life of the radionuclide

        # Calculate the time difference between injection and acquisition
        fmt = "%H%M%S.%f" if '.' in injection_time else "%H%M%S"
        inj_time = datetime.strptime(injection_time, fmt)
        acq_time = datetime.strptime(acquisition_time, fmt)
        delay_inj_acq = (acq_time - inj_time).total_seconds()

        return delay_inj_acq, injected_dose, patient_weight, nFrames, nSlices
    except Exception as e:
        raise RuntimeError(f"Error extracting dose from {dicom_file}: {e}")

def extract_acquisition_delay(dicom_file: Path) -> float:
    """
    Extracts the delay between radiopharmaceutical injection and image acquisition.

    Args:
        dicom_file (Path): Path to the DICOM file.

    Returns:
        float: The acquisition delay in seconds.
    """
    ds = pydicom.dcmread(dicom_file, stop_before_pixels=True)
    try:
        injection_time = ds.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime
        acquisition_time = ds.AcquisitionTime
        fmt = "%H%M%S.%f" if '.' in injection_time else "%H%M%S"
        inj_time = datetime.strptime(injection_time, fmt)
        acq_time = datetime.strptime(acquisition_time, fmt)
        delay_inj_acq = (acq_time - inj_time).total_seconds()
        return delay_inj_acq
    except Exception as e:
        raise RuntimeError(f"Error extracting delay from {dicom_file}: {e}")

def convert_dicom_to_nifti(dcm2niix_path: str, dicom_dir: Path, output_dir: Path, pt_id: str) -> list[Path]:
    """
    Converts a set of DICOM images into a single NIfTI file using dcm2niix.

    Args:
        dcm2niix_path (str): Path to the dcm2niix executable.
        dicom_dir (Path): Directory containing the DICOM files.
        output_dir (Path): Directory to save the output NIfTI file.
        pt_id (str): Patient ID for the output filename.

    Returns:
        list[Path]: A list containing the path to the converted NIfTI file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{pt_id}"
    cmd = [
        dcm2niix_path,
        "-b", "n",
        "-z", "n",
        "-f", fname,
        "-o", str(output_dir),
        "-s", "y",
        "-m", "y",
        str(dicom_dir)
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    nii_path = output_dir / f"{fname}.nii"
    return [nii_path]

def choose_flip_axes(file: Path, ref_orient: tuple) -> tuple:
    """
    Determines which axes to flip to match a reference orientation.

    Args:
        file (Path): Path to the NIfTI image to check.
        ref_orient (tuple): The reference orientation tuple (e.g., ('R', 'A', 'S')).

    Returns:
        tuple: A tuple (flip_x, flip_y, flip_z) with boolean values.
    """
    img = nib.load(file)
    orient = aff2axcodes(img.affine)
    flip_x = orient[0] != ref_orient[0]
    flip_y = orient[1] != ref_orient[1]
    flip_z = orient[2] != ref_orient[2]
    return flip_x, flip_y, flip_z

# --- PATLAK ANALYSIS FUNCTION ---

def pet_patlak(
    nifti_files: list[Path],
    frame_start_times: np.ndarray,
    input_function_file: Path,
    output_dir: Path,
    aif_mask_mean_values: np.ndarray
):
    """
    Performs voxel-wise Patlak analysis on dynamic PET data.

    This function calculates the Ki and Vd maps using a linear regression model
    on the linearized Patlak plot. It uses a population-based input function
    (PBIF) scaled by a patient-specific Arterial Input Function (AIF) mean.

    Args:
        nifti_files (list[Path]): A list of paths to the NIfTI files for each PET frame.
        frame_start_times (np.ndarray): Array of start times for each PET frame (in seconds).
        input_function_file (Path): Path to the population-based input function CSV file.
        output_dir (Path): Directory to save the output Ki and Vd maps.
        aif_mask_mean_values (np.ndarray): Mean activity values from the AIF mask for each frame.
    """
    n_frames = len(nifti_files)
    if len(frame_start_times) != n_frames:
        raise ValueError("Number of PET images does not match the number of time points.")

    # Convert frame start times from seconds to minutes for calculations
    frames_start_min = frame_start_times / 60

    # Load and process the population-based input function (PBIF)
    input_data = np.genfromtxt(input_function_file, delimiter=',', skip_header=1, ndmin=2)
    input_time_min = input_data[:, 0] / 60  # PBIF time in minutes
    pbif = input_data[:, 1]  # PBIF concentration

    # Scale the PBIF using the patient's AIF data
    start_index = np.searchsorted(input_time_min, frames_start_min[0], side='left')
    aif_mean = np.mean(aif_mask_mean_values)
    pbif_mean = np.mean(pbif[start_index:])
    scale_factor = aif_mean / pbif_mean
    pbif_corrected = pbif * scale_factor
    pbif_at_frame = np.interp(frames_start_min, input_time_min, pbif_corrected)

    # Calculate the cumulative integral of the corrected PBIF
    cum_int = np.concatenate(([0], cumulative_trapezoid(pbif_corrected, input_time_min)))
    int_pbif = np.interp(frames_start_min, input_time_min, cum_int)

    # Compute Patlak X values: integrated PBIF / PBIF at frame time
    dataX = int_pbif / pbif_at_frame

    # Load PET dynamic images into a single 4D array
    first_image = nib.load(nifti_files[0])
    dims = first_image.shape
    Y = np.empty((dims[0], dims[1], dims[2], n_frames))
    for i, fname in enumerate(nifti_files):
        img = nib.load(fname)
        Y[..., i] = np.squeeze(img.get_fdata())

    # Pre-allocate maps for Ki and Vd
    Ki_map = np.zeros(dims)
    Vd_map = np.zeros(dims)
    
    # Calculate Patlak X and Patlak Y for each voxel
    patlakX = np.stack([np.ones(dims[:3]) * d for d in dataX], axis=-1)
    pbif_at_frametime_stack = np.stack([np.ones(dims[:3]) * d for d in pbif_at_frame], axis=-1)
    patlakY = Y / pbif_at_frametime_stack

    # Perform voxel-wise linear regression
    # Using numpy's vectorized operations for efficiency
    meanX = np.mean(dataX)
    varX = np.var(dataX)
    meanY = np.mean(patlakY, axis=-1)
    covXY = np.mean(patlakY * patlakX, axis=-1) - meanY * meanX

    slope = covXY / varX
    intercept = meanY - slope * meanX

    # Assign calculated values to the maps
    Ki_map = slope * 100 # Convert ml/min/ml to ml/min/100ml
    Vd_map = intercept * 100  # Convert Vd to percentage

    # Set negative values to zero
    Ki_map[Ki_map < 0] = 0
    Vd_map[Vd_map < 0] = 0

    # Save the output maps as NIFTI files
    affine = first_image.affine
    imgKi = nib.Nifti1Image(Ki_map.astype(np.float32), affine)
    imgVd = nib.Nifti1Image(Vd_map.astype(np.float32), affine)
    
    nib.save(imgKi, os.path.join(output_dir, 'Ki_map.nii'))
    nib.save(imgVd, os.path.join(output_dir, 'Vd_map.nii'))

    print(f"Saved Ki map to: {os.path.join(output_dir, 'Ki_map.nii')}")

    print(f"Saved Vd map to: {os.path.join(output_dir, 'Vd_map.nii')}")

