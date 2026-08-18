# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""
Pre-align multi-session BIDS data to a reference session.

For each non-reference session, registers the session T1w to the reference
session T1w (rigid, 6-DOF) using FreeSurfer's ``mri_robust_register``, then
injects the resulting RAS transform into the NIfTI headers of all functional
and fieldmap images.  No voxel data is resampled — only the affine is updated.

After pre-alignment, fMRIPrep's coregistration (``mri_coreg`` / ``bbregister``)
only needs to refine a small residual instead of bridging a large cross-session
displacement.

Usage::

    fmriprep-prealign /data/bids /data/bids_prealigned \\
        --participant-label 1181004 \\
        --ref-session 1
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

logger = logging.getLogger('fmriprep.prealign')


# ---------------------------------------------------------------------------
# LTA helpers
# ---------------------------------------------------------------------------

def load_lta_ras(lta_path: str | Path) -> np.ndarray:
    """Read a FreeSurfer LTA file and return the RAS-to-RAS 4×4 matrix.

    Only type-1 (RAS2RAS) LTA files are supported.  ``mri_robust_register``
    writes this format by default.
    """
    with open(lta_path) as fobj:
        lines = fobj.readlines()

    # Determine LTA type (0 = VOX2VOX, 1 = RAS2RAS)
    lta_type = 1  # default
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('type'):
            # Handle "type = 1 # LINEAR_RAS_TO_RAS" style comments
            val = stripped.split('=')[-1].strip().rstrip(';')
            val = val.split('#')[0].strip()  # strip inline comments
            lta_type = int(val)
            break

    if lta_type != 1:
        raise ValueError(
            f'Only RAS2RAS (type 1) LTA files are supported, got type {lta_type} '
            f'in {lta_path}'
        )

    # Parse the 4×4 matrix (appears after a line reading "1 4 4")
    mat_rows: list[list[float]] = []
    reading = False
    for line in lines:
        if line.strip() == '1 4 4':
            reading = True
            continue
        if reading:
            vals = line.strip().split()
            if len(vals) == 4:
                mat_rows.append([float(v) for v in vals])
            if len(mat_rows) == 4:
                break

    if len(mat_rows) != 4:
        raise ValueError(f'Could not parse 4×4 matrix from {lta_path}')

    return np.array(mat_rows)


def inject_affine(
    nifti_path: str | Path,
    xfm_inv: np.ndarray,
    backup_suffix: str = '_origSpace',
) -> None:
    """Write ``inv(xfm) @ original_affine`` into *nifti_path*'s header.

    A backup of the original file is saved with *backup_suffix* appended
    before the extension.  If the backup already exists it is reused as the
    source (idempotent).

    Parameters
    ----------
    nifti_path : path
        Path to the NIfTI file to modify (overwritten in-place).
    xfm_inv : (4, 4) array
        Inverse of the session-to-reference RAS transform.
    backup_suffix : str
        Suffix inserted before ``.nii`` / ``.nii.gz`` for the backup.
    """
    nifti_path = str(nifti_path)

    # Build backup path
    if nifti_path.endswith('.nii.gz'):
        backup = nifti_path.replace('.nii.gz', f'{backup_suffix}.nii.gz')
    else:
        backup = nifti_path.replace('.nii', f'{backup_suffix}.nii')

    if not os.path.exists(backup):
        os.rename(nifti_path, backup)

    img = nib.load(backup)
    new_affine = xfm_inv @ img.affine

    new_img = img.__class__(img.dataobj, affine=new_affine, header=img.header)
    new_img.header.set_sform(new_affine)
    new_img.header.set_qform(new_affine)
    nib.save(new_img, nifti_path)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def register_session(
    mov_t1w: str | Path,
    ref_t1w: str | Path,
    out_lta: str | Path,
    out_mapped: str | Path | None = None,
) -> None:
    """Run ``mri_robust_register`` (rigid, 6-DOF) between two T1w images."""
    cmd = [
        'mri_robust_register',
        '--mov', str(mov_t1w),
        '--dst', str(ref_t1w),
        '--lta', str(out_lta),
        '--satit',
        '--iscale',
        '--affine',
    ]
    if out_mapped is not None:
        cmd += ['--mapmov', str(out_mapped)]

    logger.info('Running: %s', ' '.join(cmd))
    subprocess.run(cmd, check=True)


def prealign_session(
    ses_dir: str | Path,
    ref_t1w: str | Path,
    subject: str,
    ses_label: str,
) -> None:
    """Register one session's T1w to the reference and inject the transform.

    Parameters
    ----------
    ses_dir : path
        Absolute path to ``<bids_out>/<subject>/ses-<label>/``
        (already copied; will be modified in-place).
    ref_t1w : path
        Path to the reference-session T1w.
    subject : str
        BIDS subject label (with ``sub-`` prefix).
    ses_label : str
        BIDS session label (without ``ses-`` prefix).
    """
    ses_dir = Path(ses_dir)

    # Find session T1w
    t1w_candidates = sorted(ses_dir.glob('anat/*_T1w.nii.gz'))
    if not t1w_candidates:
        logger.warning('No T1w found for ses-%s, skipping', ses_label)
        return
    mov_t1w = t1w_candidates[0]

    # Output paths
    lta_file = ses_dir / 'anat' / f'{subject}_ses-{ses_label}_from-sesN_to-ses1_xfm.lta'
    mapped_file = ses_dir / 'anat' / f'{subject}_ses-{ses_label}_space-ses1_T1w.nii.gz'

    logger.info('Registering ses-%s T1w → reference T1w ...', ses_label)
    register_session(mov_t1w, ref_t1w, lta_file, mapped_file)
    logger.info('LTA saved: %s', lta_file)

    # Load transform and compute inverse
    xfm = load_lta_ras(lta_file)
    xfm_inv = np.linalg.inv(xfm)

    # Collect all NIfTI files in func/ and fmap/, excluding _origSpace backups
    nifti_files: list[str] = []
    for subdir in ('func', 'fmap'):
        for f in sorted(glob.glob(str(ses_dir / subdir / '*.nii.gz'))):
            if '_origSpace' not in f:
                nifti_files.append(f)
        for f in sorted(glob.glob(str(ses_dir / subdir / '*.nii'))):
            if '_origSpace' not in f:
                nifti_files.append(f)

    # If _origSpace files exist from a previous pre-alignment, restore the
    # originals first so we always transform from true scanner coordinates.
    for f in nifti_files:
        if f.endswith('.nii.gz'):
            orig = f.replace('.nii.gz', '_origSpace.nii.gz')
        else:
            orig = f.replace('.nii', '_origSpace.nii')
        if os.path.exists(orig):
            logger.info('  Restoring original scanner-space header from %s', os.path.basename(orig))
            shutil.copy2(orig, f)

    if not nifti_files:
        logger.warning('No func/fmap NIfTI files in ses-%s', ses_label)
        return

    logger.info('Injecting RAS transform into %d NIfTI headers ...', len(nifti_files))
    for f in nifti_files:
        inject_affine(f, xfm_inv)
        logger.info('  ✓ %s', os.path.basename(f))

    logger.info('ses-%s complete (%d files updated).', ses_label, len(nifti_files))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Pre-align multi-session BIDS data to a reference session.  '
            'Registers each session\'s T1w to the reference via '
            'mri_robust_register (rigid 6-DOF), then writes the resulting '
            'RAS transform into all func/fmap NIfTI headers (no resampling).'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'bids_dir',
        type=Path,
        help='Input BIDS directory.',
    )
    parser.add_argument(
        'output_dir',
        type=Path,
        help='Output directory for the pre-aligned BIDS dataset.  '
        'The reference session is copied verbatim; other sessions are '
        'copied and their func/fmap headers updated.',
    )
    parser.add_argument(
        '--participant-label',
        required=True,
        help='BIDS participant label (without "sub-" prefix).',
    )
    parser.add_argument(
        '--ref-session',
        default='1',
        help='Reference session label (without "ses-" prefix).  '
        'All other sessions are registered to this one.',
    )
    parser.add_argument(
        '--sessions',
        nargs='*',
        default=None,
        help='Session labels to pre-align (without "ses-" prefix).  '
        'Default: all sessions except the reference.',
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging.',
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(levelname)s: %(message)s',
    )

    bids_dir = args.bids_dir.resolve()
    output_dir = args.output_dir.resolve()
    subject = f'sub-{args.participant_label}'
    ref_ses = args.ref_session

    sub_dir = bids_dir / subject
    if not sub_dir.is_dir():
        logger.error('Subject directory not found: %s', sub_dir)
        sys.exit(1)

    # Discover sessions
    all_sessions = sorted(
        d.name.replace('ses-', '')
        for d in sub_dir.iterdir()
        if d.is_dir() and d.name.startswith('ses-')
    )
    if ref_ses not in all_sessions:
        logger.error('Reference session ses-%s not found in %s', ref_ses, sub_dir)
        sys.exit(1)

    sessions_to_align = args.sessions or [s for s in all_sessions if s != ref_ses]
    logger.info(
        'Reference session: ses-%s | Sessions to align: %s',
        ref_ses,
        ', '.join(f'ses-{s}' for s in sessions_to_align),
    )

    # Prepare output directory
    out_sub = output_dir / subject
    out_sub.mkdir(parents=True, exist_ok=True)

    # Copy dataset_description.json
    dd = bids_dir / 'dataset_description.json'
    if dd.exists():
        shutil.copy2(dd, output_dir / 'dataset_description.json')

    # Copy reference session verbatim
    ref_out = out_sub / f'ses-{ref_ses}'
    if not ref_out.exists():
        logger.info('Copying reference session ses-%s ...', ref_ses)
        shutil.copytree(
            sub_dir / f'ses-{ref_ses}',
            ref_out,
            symlinks=False,     # dereference symlinks
            dirs_exist_ok=True,
        )

    # Find reference T1w
    ref_t1w_candidates = sorted(ref_out.glob('anat/*_T1w.nii.gz'))
    if not ref_t1w_candidates:
        logger.error('No T1w found in reference session ses-%s', ref_ses)
        sys.exit(1)
    ref_t1w = ref_t1w_candidates[0]
    logger.info('Reference T1w: %s', ref_t1w)

    # Process each session
    for ses in sessions_to_align:
        ses_src = sub_dir / f'ses-{ses}'
        if not ses_src.is_dir():
            logger.warning('ses-%s not found, skipping', ses)
            continue

        ses_out = out_sub / f'ses-{ses}'
        if not ses_out.exists():
            logger.info('Copying ses-%s ...', ses)
            shutil.copytree(ses_src, ses_out, symlinks=False, dirs_exist_ok=True)

        prealign_session(ses_out, ref_t1w, subject, ses)

    logger.info('Pre-alignment complete.  Output: %s', output_dir)
    logger.info(
        'Next: run fmriprep with --bold2anat-init header --bbr-hemi <lh|rh> '
        '--coreg-regheader --coreg-no-brute-force on the pre-aligned data.'
    )


if __name__ == '__main__':
    main()
