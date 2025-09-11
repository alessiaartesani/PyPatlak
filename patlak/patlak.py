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
import argparse
import shutil



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

# Convert DICOMs for each frame to a separate NIfTI file (no 4D NIfTI)
def convert_dicom_frames_to_nifti(dcm2niix_path: str, dicom_files: list, output_dir: Path, pt_id: str, n_frames: int, n_slices: int) -> list:
    output_dir.mkdir(parents=True, exist_ok=True)
    nifti_files = []
    for frame_idx in range(n_frames):
        batch_files = dicom_files[frame_idx * n_slices : (frame_idx + 1) * n_slices]
        temp_frame_folder = output_dir / f"temp_frame_{frame_idx:03d}"
        temp_frame_folder.mkdir(parents=True, exist_ok=True)
        for f in batch_files:
            shutil.copy(f, temp_frame_folder)
        fname = f"{pt_id}_frame_{frame_idx:03d}"
        cmd = [
            dcm2niix_path,
            "-b", "n",
            "-z", "n",
            "-f", fname,
            "-o", str(output_dir),
            "-s", "y",
            "-m", "y",
            str(temp_frame_folder)
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        nii_path = output_dir / f"{fname}.nii"
        nifti_files.append(nii_path)
        shutil.rmtree(temp_frame_folder)
    return nifti_files


    # Deprecated: Not used in current workflow. Kept for reference.
    pass

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

    # Use the middle of each frame for scaling (more robust)
    # Calculate middle of frame times in minutes
    if len(frames_start_min) > 1:
        frame_duration_min = np.diff(frames_start_min)
        frame_duration_min = np.append(frame_duration_min, frame_duration_min[-1])
    else:
        frame_duration_min = np.array([1.0])  # fallback if only one frame
    middle_frame_time = frames_start_min + frame_duration_min / 2

    # Calculate AUC for AIF and PBIF at middle frame times
    auc_aif_patient = np.trapezoid(aif_mask_mean_values, x=middle_frame_time)
    pbif_at_frame = np.interp(middle_frame_time, input_time_min, pbif)
    auc_pbif = np.trapezoid(pbif_at_frame, x=middle_frame_time)
    scale_factor = auc_aif_patient / auc_pbif
    pbif_corrected = pbif * scale_factor
    pbif_at_frame = np.interp(middle_frame_time, input_time_min, pbif_corrected)

    # Calculate the cumulative integral of the corrected PBIF
    cum_int = np.concatenate(([0], cumulative_trapezoid(pbif_corrected, input_time_min)))
    int_pbif = np.interp(middle_frame_time, input_time_min, cum_int)

    # Compute Patlak X values: integrated PBIF / PBIF at frame time (both at middle_frame_time)
    dataX = int_pbif / pbif_at_frame


    # Load PET dynamic images into a single 4D array (x, y, z, time)
    first_image = nib.load(nifti_files[0])
    dims = first_image.shape
    Y = np.empty((dims[0], dims[1], dims[2], n_frames))
    for i, fname in enumerate(nifti_files):
        img = nib.load(fname)
        Y[..., i] = np.squeeze(img.get_fdata())

    # Optionally remove NIfTI files after loading (uncomment if needed)
    for fname in nifti_files:
        if Path(fname).exists():
            Path(fname).unlink()

    # Pre-allocate maps for Ki and Vd
    Ki_map = np.zeros(dims)
    Vd_map = np.zeros(dims)

    # Prepare Patlak X for regression
    patlakX = np.stack([np.ones((dims[0], dims[1])) * d for d in dataX], axis=-1)

    # Voxel-wise Patlak regression for each slice
    for iSlice in range(dims[2]):
        sliceData = Y[:, :, iSlice, :]
        pbif_at_frametime_stack = np.stack([np.ones((dims[0], dims[1])) * d for d in pbif_at_frame], axis=-1)
        patlakY = sliceData / pbif_at_frametime_stack

        meanX = np.mean(dataX)
        varX = np.mean((dataX - meanX)**2)
        meanY = np.mean(patlakY, axis=2)
        covXY = np.mean(patlakY * patlakX, axis=2) - meanY * meanX
        slope = covXY / varX
        intercept = meanY - slope * meanX

        Ki_map[:, :, iSlice] = slope.reshape(dims[0], dims[1]) * 100 # Convert Ki to ml/min/100ml
        Vd_map[:, :, iSlice] = intercept.reshape(dims[0], dims[1]) * 100  # Convert Vd to percentage

    # Set negative values to zero (non-physiological)
    Ki_map[Ki_map < 0] = 0
    Vd_map[Vd_map < 0] = 0

    # Save the output maps as NIfTI files
    affine = first_image.affine
    imgKi = nib.Nifti1Image(Ki_map.astype(np.float32), affine)
    imgVd = nib.Nifti1Image(Vd_map.astype(np.float32), affine)
    nib.save(imgKi, os.path.join(output_dir, 'Ki_map.nii'))
    nib.save(imgVd, os.path.join(output_dir, 'Vd_map.nii'))

    print(f"Saved Ki map to: {os.path.join(output_dir, 'Ki_map.nii')}")
    print(f"Saved Vd map to: {os.path.join(output_dir, 'Vd_map.nii')}")


def main():
    """
    Main function to run the Patlak analysis from the command line.
    """
    parser = argparse.ArgumentParser(description="Perform Patlak analysis on dynamic PET data.")
    parser.add_argument("dicom_dir", type=str, help="Path to the directory containing DICOM files.")
    parser.add_argument("aif_mask_path", type=str, help="Path to the AIF mask NIfTI file (.nii or .nii.gz).")
    parser.add_argument("output_dir", type=str, help="Path to the output directory for results.")
    parser.add_argument("pt_id", type=str, help="Patient ID for naming output files.")
    parser.add_argument("--rf", type=float, default=1.0,
                        help="Recovery factor to scale AIF mask mean values (default: 1.0)")
    parser.add_argument("--input_function", type=str, default="./input_function.csv",
                        help="Path to the population-based input function CSV file.")
    parser.add_argument("--dcm2niix_path", type=str, default="dcm2niix",
                        help="Path to the dcm2niix executable. Assumes it's in PATH by default.")


    args = parser.parse_args()

    dicom_dir = Path(args.dicom_dir)
    aif_mask_path = Path(args.aif_mask_path)
    output_dir = Path(args.output_dir)
    pt_id = args.pt_id
    input_function_file = Path(args.input_function)
    dcm2niix_path = args.dcm2niix_path

    # Check if necessary directories and files exist
    if not dicom_dir.is_dir():
        raise FileNotFoundError(f"DICOM directory not found: {dicom_dir}")
    if not aif_mask_path.is_file():
        raise FileNotFoundError(f"AIF mask file not found: {aif_mask_path}")
    if not input_function_file.is_file():
        raise FileNotFoundError(f"Input function CSV not found: {input_function_file}")

    print(f"=========== STARTING PATLAK ANALYSIS FOR PATIENT: {pt_id} ===========")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)


    # Delete old output folder if it exists (removes all previous results)
    if output_dir.exists():
        shutil.rmtree(output_dir)


    # Convert DICOM files for each frame to a separate NIfTI file (no 4D NIfTI)
    print("[✓] Converting DICOMs to NIfTI (one file per frame)...")
    dicom_files = sorted(list(dicom_dir.rglob("*.dcm")))
    if not dicom_files:
        raise FileNotFoundError(f"No DICOM files found in {dicom_dir}")
    first_dicom = dicom_files[0]
    inj_delay, _, _, n_frames, n_slices = extract_dicom_info(first_dicom)
    nifti_files_path = [output_dir / f"{pt_id}_frame_{i:03d}.nii" for i in range(n_frames)]
    if all(f.exists() for f in nifti_files_path):
        print("[✓] NIfTI files for all frames already exist. Skipping conversion.")
    else:
        nifti_files_path = convert_dicom_frames_to_nifti(dcm2niix_path, dicom_files, output_dir, pt_id, n_frames, n_slices)
    if not nifti_files_path or not all(f.exists() for f in nifti_files_path):
        raise FileNotFoundError(f"Not all NIfTI files were created in {output_dir}")


    # Extract frame timings from DICOM headers
    delay_slice = np.array([extract_acquisition_delay(d) for d in dicom_files])
    delay_frame = np.array([delay_slice[i * n_slices] for i in range(n_frames)])
    print(f"[✓] Delay between injection and acquisition frame: {delay_frame} s")
    frame_duration = np.diff(delay_frame)
    frame_duration = np.append(frame_duration, frame_duration[-1])
    print(f"[✓] Frame durations: {frame_duration} s")
    middle_frame_time = delay_frame + frame_duration / 2
    print(f"[✓] Middle frame times: {middle_frame_time} s")


    # Decompress AIF mask if necessary and load it
    print("[✓] Loading and preparing AIF mask...")
    if str(aif_mask_path).endswith('.gz'):
        img = nib.load(aif_mask_path)
        decompressed_file_path = str(aif_mask_path)[:-3]  # remove '.gz'
        nib.save(img, decompressed_file_path)
        print(f"[✓] Decompressed Mask file and saved to {decompressed_file_path}")
        aif_mask_path = decompressed_file_path

    aif_mask_img = nib.load(aif_mask_path)
    aif_mask_data = aif_mask_img.get_fdata()
    aif_orient = aff2axcodes(aif_mask_img.affine)

    # Reorient AIF mask to match PET image orientation if needed
    pet_orient = aff2axcodes(nib.load(nifti_files_path[0]).affine)
    if pet_orient != aif_orient:
        print("[!] AIF Mask orientation does not match NIfTI orientation. Flipping axes...")
        flip_x, flip_y, flip_z = choose_flip_axes(nifti_files_path[0], aif_orient)
        if flip_x:
            aif_mask_data = np.flip(aif_mask_data, axis=0)
        if flip_y:
            aif_mask_data = np.flip(aif_mask_data, axis=1)
        if flip_z:
            aif_mask_data = np.flip(aif_mask_data, axis=2)
        print("[✓] AIF Mask reoriented.")


    # Extract mean values from PET frames using the AIF mask
    aif_mask_mean = []
    for nifti_file in nifti_files_path:
        nifti_img = nib.load(nifti_file)
        nifti_data = nifti_img.get_fdata()
        masked_data = nifti_data[aif_mask_data > 0]
        if masked_data.size > 0:
            mean_value = np.mean(masked_data)
            aif_mask_mean.append(mean_value)

    if not aif_mask_mean:
        raise ValueError("AIF mask is empty or does not overlap with PET data.")

    # Apply recovery factor to AIF mask mean values
    recovery_factor = args.rf
    aif_mask_mean = np.array(aif_mask_mean) * recovery_factor
    print(f"[✓] AIF Mask Mean Values (recovery_factor={recovery_factor}): {aif_mask_mean}")


    # Run Patlak analysis and save results
    print("[✓] Performing Patlak analysis...")
    pet_patlak(nifti_files_path, delay_frame, input_function_file, output_dir, aif_mask_mean)
    print("============================= PATLAK ANALYSIS COMPLETE =============================")



