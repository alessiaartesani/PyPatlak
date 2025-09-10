import argparse
import os
from pathlib import Path
import numpy as np
import nibabel as nib
from nibabel.orientations import aff2axcodes
from .patlak import (
    extract_dicom_info,
    extract_acquisition_delay,
    convert_dicom_to_nifti,
    pet_patlak,
    choose_flip_axes
)

def main():
    """
    Main function to run the Patlak analysis from the command line.
    """
    parser = argparse.ArgumentParser(description="Perform Patlak analysis on dynamic PET data.")
    parser.add_argument("dicom_dir", type=str, help="Path to the directory containing DICOM files.")
    parser.add_argument("aif_mask_path", type=str, help="Path to the AIF mask NIfTI file (.nii or .nii.gz).")
    parser.add_argument("output_dir", type=str, help="Path to the output directory for results.")
    parser.add_argument("pt_id", type=str, help="Patient ID for naming output files.")
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

    print("=====================================================================")
    print(f"=========== STARTING PATLAK ANALYSIS FOR PATIENT: {pt_id} ===========")
    print("=====================================================================")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert DICOM files to a single NIfTI file (dynamic)
    print("[✓] Converting DICOMs to NIfTI...")
    nifti_files_path = convert_dicom_to_nifti(dcm2niix_path, dicom_dir, output_dir, pt_id)

    # Extract frame timings from DICOM headers
    dicom_files = sorted(list(dicom_dir.rglob("*.dcm")))
    if not dicom_files:
        raise FileNotFoundError(f"No DICOM files found in {dicom_dir}")
    first_dicom = dicom_files[0]
    inj_delay, _, _, n_frames, n_slices = extract_dicom_info(first_dicom)

    delay_slice = np.array([extract_acquisition_delay(d) for d in dicom_files])
    delay_frame = np.array([delay_slice[i * n_slices] for i in range(n_frames)])
    print(f"[✓] Extracted frame delays: {delay_frame} s")

    # Decompress AIF mask if necessary and load it
    print("[✓] Loading and preparing AIF mask...")
    if aif_mask_path.suffix == '.gz':
        img = nib.load(aif_mask_path)
        decompressed_file_path = aif_mask_path.with_suffix('')
        nib.save(img, decompressed_file_path)
        aif_mask_path = decompressed_file_path
    
    aif_mask_img = nib.load(aif_mask_path)
    aif_mask_data = aif_mask_img.get_fdata()
    aif_orient = aff2axcodes(aif_mask_img.affine)

    # Check and reorient AIF mask to match PET image orientation
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
        # Apply the AIF mask
        masked_data = nifti_data[aif_mask_data > 0]
        if masked_data.size > 0:
            mean_value = np.mean(masked_data)
            aif_mask_mean.append(mean_value)
    
    if not aif_mask_mean:
        raise ValueError("AIF mask is empty or does not overlap with PET data.")
    
    aif_mask_mean = np.array(aif_mask_mean)
    print(f"[✓] AIF Mask Mean Values: {aif_mask_mean}")

    # Perform Patlak analysis
    print("[✓] Performing Patlak analysis...")
    pet_patlak(nifti_files_path, delay_frame, input_function_file, output_dir, aif_mask_mean)

    print("=====================================================================")
    print("=========== PATLAK ANALYSIS COMPLETE ================================")
    print("=====================================================================")

if __name__ == "__main__":

    main()
