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
"""Pre-align the sessions of a BIDS subject to a reference session.

Each session's T1w is registered (rigid) to the reference session's T1w with
``mri_robust_register``, and the transform is written into the headers of all NIfTI
images of that session in a copy of the dataset. Voxel data are not modified.
Registration outputs are written to ``<output_dir>/sourcedata/prealign/``.
"""

from __future__ import annotations

import argparse
import gzip
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import nibabel as nb
import numpy as np

logger = logging.getLogger('fmriprep.prealign')


def _is_nifti(path: str | Path) -> bool:
    return str(path).endswith(('.nii', '.nii.gz'))


def _find_t1w(ses_dir: Path) -> Path | None:
    candidates = sorted(path for path in (ses_dir / 'anat').glob('*_T1w.nii*') if _is_nifti(path))
    return candidates[0] if candidates else None


def load_lta_ras2ras(lta_path: str | Path) -> np.ndarray:
    """Return the RAS-to-RAS (source to destination) matrix of an LTA file."""
    from nitransforms.io.lta import FSLinearTransformArray

    xfm = FSLinearTransformArray.from_filename(lta_path)['xforms'][0]
    xfm.set_type(1)
    return np.array(xfm['m_L'], dtype=float)


def max_displacement(xfm: np.ndarray, radius: float = 80.0) -> float:
    """Largest displacement (mm) of the corners of a head-sized cube under ``xfm``."""
    corners = np.array(np.meshgrid(*[[-radius, radius]] * 3)).reshape(3, -1)
    corners = np.vstack((corners, np.ones(corners.shape[1])))
    return float(np.max(np.linalg.norm((xfm @ corners - corners)[:3], axis=0)))


def _similarity(ref: nb.Nifti1Image, mov: nb.Nifti1Image, affine: np.ndarray) -> float:
    from nibabel.processing import resample_from_to

    factor = max(1, round(3.0 / min(ref.header.get_zooms()[:3])))
    grid_affine = ref.affine @ np.diag([factor, factor, factor, 1])
    grid = (tuple(int(np.ceil(n / factor)) for n in ref.shape[:3]), grid_affine)

    ref_data = resample_from_to(ref, grid, order=1).get_fdata()
    moved = nb.Nifti1Image(np.asanyarray(mov.dataobj, dtype=np.float32), affine)
    mov_data = resample_from_to(moved, grid, order=1).get_fdata()

    overlap = (ref_data > 0) & (mov_data > 0)
    if overlap.sum() < 100:
        return -1.0
    return float(np.corrcoef(ref_data[overlap], mov_data[overlap])[0, 1])


def check_direction(mov_t1w, ref_t1w, xfm: np.ndarray, min_displacement: float = 2.0) -> None:
    """Raise if the inverse of ``xfm`` aligns ``mov_t1w`` to ``ref_t1w`` better than ``xfm``."""
    if max_displacement(xfm) < min_displacement:
        return

    ref, mov = nb.load(ref_t1w), nb.load(mov_t1w)
    forward = _similarity(ref, mov, xfm @ mov.affine)
    inverse = _similarity(ref, mov, np.linalg.inv(xfm) @ mov.affine)
    logger.info('T1w correlation with the reference: %.3f (inverse: %.3f)', forward, inverse)
    if inverse > forward:
        raise RuntimeError(
            f'The inverse of the registration aligns {mov_t1w} to {ref_t1w} better '
            'than the registration itself; not writing headers.'
        )


def rewrite_affine(in_file: str | Path, out_file: str | Path, xfm: np.ndarray) -> np.ndarray:
    """Copy a NIfTI file with its affine premultiplied by ``xfm``; return the new affine."""
    in_file, out_file = str(in_file), str(out_file)
    header_class = type(nb.load(in_file).header)
    in_opener = gzip.open if in_file.endswith('.gz') else open

    with in_opener(in_file, 'rb') as fin:
        header = header_class(fin.read(header_class.sizeof_hdr), check=False)
        new_affine = xfm @ header.get_best_affine()

        set_sform = int(header['sform_code']) > 0
        set_qform = int(header['qform_code']) > 0
        if set_sform or not set_qform:
            header.set_sform(new_affine)
        if set_qform or not set_sform:
            header.set_qform(new_affine)

        # Extensions and voxel data are copied byte-for-byte
        out_opener = gzip.open if out_file.endswith('.gz') else open
        with out_opener(out_file, 'wb') as fout:
            fout.write(header.binaryblock)
            shutil.copyfileobj(fin, fout, length=1 << 24)

    return new_affine


def register_session(mov_t1w, ref_t1w, out_lta, out_mapped=None) -> None:
    cmd = [
        'mri_robust_register',
        '--mov',
        str(mov_t1w),
        '--dst',
        str(ref_t1w),
        '--lta',
        str(out_lta),
        '--satit',
        '--iscale',
    ]
    if out_mapped is not None:
        cmd += ['--mapmov', str(out_mapped)]
    logger.info('Running: %s', ' '.join(cmd))
    subprocess.run(cmd, check=True)


def prealign_session(in_ses_dir: Path, out_ses_dir: Path, ref_t1w: Path, work_dir: Path) -> None:
    """Register a session's T1w to ``ref_t1w`` and write all its NIfTIs to ``out_ses_dir``."""
    mov_t1w = _find_t1w(in_ses_dir)
    if mov_t1w is None:
        raise RuntimeError(f'No T1w found in {in_ses_dir / "anat"}')

    work_dir.mkdir(parents=True, exist_ok=True)
    lta_file = work_dir / 'from-session_to-reference_mode-image_xfm.lta'
    register_session(mov_t1w, ref_t1w, lta_file, work_dir / 'space-reference_T1w.nii.gz')

    xfm = load_lta_ras2ras(lta_file)
    logger.info(
        'Session-to-reference transform (max. displacement %.1f mm):\n%s',
        max_displacement(xfm),
        np.array2string(xfm, precision=4, suppress_small=True),
    )
    check_direction(mov_t1w, ref_t1w, xfm)

    for in_file in sorted(path for path in in_ses_dir.rglob('*') if _is_nifti(path)):
        out_file = out_ses_dir / in_file.relative_to(in_ses_dir)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        rewrite_affine(in_file, out_file, xfm)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('bids_dir', type=Path, help='Input BIDS directory (not modified).')
    parser.add_argument('output_dir', type=Path, help='Output directory.')
    parser.add_argument('--participant-label', required=True, help='Label without "sub-".')
    parser.add_argument('--ref-session', default='1', help='Reference session, without "ses-".')
    parser.add_argument(
        '--sessions', nargs='*', help='Sessions to pre-align (default: all but the reference).'
    )
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging.')
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
    if output_dir == bids_dir or bids_dir in output_dir.parents:
        parser.error('The output directory must not be the input dataset or inside it.')

    subject = f'sub-{args.participant_label.removeprefix("sub-")}'
    ref_ses = args.ref_session.removeprefix('ses-')
    sub_dir = bids_dir / subject
    if not sub_dir.is_dir():
        parser.error(f'Subject directory not found: {sub_dir}')

    all_sessions = sorted(
        d.name.removeprefix('ses-')
        for d in sub_dir.iterdir()
        if d.is_dir() and d.name.startswith('ses-')
    )
    if ref_ses not in all_sessions:
        parser.error(f'Reference session ses-{ref_ses} not found in {sub_dir}')

    sessions = [s.removeprefix('ses-') for s in (args.sessions or all_sessions)]
    sessions = [s for s in sessions if s != ref_ses]
    if missing := sorted(set(sessions) - set(all_sessions)):
        parser.error(f'Session(s) not found in {sub_dir}: {", ".join(missing)}')

    ref_t1w = _find_t1w(sub_dir / f'ses-{ref_ses}')
    if ref_t1w is None:
        parser.error(f'No T1w found in reference session ses-{ref_ses}')

    # Top-level files include dataset metadata and inherited sidecars
    output_dir.mkdir(parents=True, exist_ok=True)
    for entry in bids_dir.iterdir():
        if entry.is_file():
            shutil.copy2(entry, output_dir / entry.name)

    # Copy the subject, except the NIfTI files that are rewritten below
    aligned_dirs = {sub_dir / f'ses-{ses}' for ses in sessions}

    def _ignore(directory, names):
        directory = Path(directory)
        if directory in aligned_dirs or any(d in directory.parents for d in aligned_dirs):
            return [name for name in names if _is_nifti(name)]
        return []

    shutil.copytree(sub_dir, output_dir / subject, ignore=_ignore, dirs_exist_ok=True)

    for ses in sessions:
        logger.info('Pre-aligning ses-%s to ses-%s', ses, ref_ses)
        prealign_session(
            sub_dir / f'ses-{ses}',
            output_dir / subject / f'ses-{ses}',
            ref_t1w,
            output_dir / 'sourcedata' / 'prealign' / subject / f'ses-{ses}',
        )


if __name__ == '__main__':
    sys.exit(main())
