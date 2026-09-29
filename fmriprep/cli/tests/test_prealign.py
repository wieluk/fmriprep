import gzip
import json

import nibabel as nb
import numpy as np
import pytest
from nitransforms.io.lta import FSLinearTransformArray, VolumeGeometry
from scipy.ndimage import gaussian_filter

from .. import prealign


def _rigid(angles, translation):
    xfm = np.eye(4)
    for axis, angle in enumerate(angles):
        c, s = np.cos(angle), np.sin(angle)
        i, j = [ax for ax in range(3) if ax != axis]
        rot = np.eye(4)
        rot[[i, i, j, j], [i, j, i, j]] = c, -s, s, c
        xfm = rot @ xfm
    xfm[:3, 3] = translation
    return xfm


# Maps session RAS onto reference RAS
SES2REF = _rigid((0.08, -0.05, 0.12), (6.0, -9.0, 4.0))
REF_AFFINE = np.array([[-2.0, 0, 0, 40], [0, 2.0, 0, -46], [0, 0, 2.0, -38], [0, 0, 0, 1]])


@pytest.fixture(scope='module')
def head():
    data = np.zeros((40, 46, 38), dtype=np.float32)
    data[6:-6, 6:-6, 5:-5] = np.random.default_rng(1234).random((28, 34, 28))
    return (1000 * gaussian_filter(data, 2)).astype(np.int16)


@pytest.mark.parametrize('ext', ['.nii', '.nii.gz'])
@pytest.mark.parametrize(('qform_code', 'sform_code'), [(1, 1), (1, 0), (0, 2)])
def test_rewrite_affine(tmp_path, head, ext, qform_code, sform_code):
    in_file, out_file = tmp_path / f'in{ext}', tmp_path / f'out{ext}'
    img = nb.Nifti1Image(head, REF_AFFINE)
    img.header.set_qform(REF_AFFINE, code=qform_code)
    img.header.set_sform(REF_AFFINE, code=sform_code)
    img.to_filename(in_file)

    prealign.rewrite_affine(in_file, out_file, SES2REF)

    out = nb.load(out_file)
    assert np.allclose(out.affine, SES2REF @ REF_AFFINE)
    assert (int(out.header['qform_code']), int(out.header['sform_code'])) == (
        qform_code,
        sform_code,
    )
    opener = gzip.open if ext == '.nii.gz' else open
    with opener(in_file, 'rb') as fin, opener(out_file, 'rb') as fout:
        assert fin.read()[348:] == fout.read()[348:]


def test_check_direction(tmp_path, head):
    ref, mov = tmp_path / 'ref.nii.gz', tmp_path / 'mov.nii.gz'
    nb.Nifti1Image(head, REF_AFFINE).to_filename(ref)
    nb.Nifti1Image(head, np.linalg.inv(SES2REF) @ REF_AFFINE).to_filename(mov)

    prealign.check_direction(mov, ref, SES2REF)
    with pytest.raises(RuntimeError, match='inverse'):
        prealign.check_direction(mov, ref, np.linalg.inv(SES2REF))


def test_prealign_main(tmp_path, head, monkeypatch):
    bids_dir, out_dir = tmp_path / 'bids', tmp_path / 'prealigned'
    sub = bids_dir / 'sub-01'
    bids_dir.mkdir()
    (bids_dir / 'dataset_description.json').write_text(json.dumps({'Name': 'x'}))

    affines = {'1': REF_AFFINE, '2': np.linalg.inv(SES2REF) @ REF_AFFINE}
    for ses, affine in affines.items():
        for datatype, suffix in (('anat', 'T1w'), ('func', 'task-rest_bold')):
            fname = sub / f'ses-{ses}' / datatype / f'sub-01_ses-{ses}_{suffix}.nii.gz'
            fname.parent.mkdir(parents=True, exist_ok=True)
            nb.Nifti1Image(head, affine).to_filename(fname)

    def _fake_register(mov_t1w, ref_t1w, out_lta, out_mapped=None):
        lta = FSLinearTransformArray.from_ras(np.linalg.inv(SES2REF)[np.newaxis])
        xfm = lta['xforms'][0]
        xfm.structarr['src'] = VolumeGeometry.from_image(nb.load(mov_t1w)).structarr
        xfm.structarr['dst'] = VolumeGeometry.from_image(nb.load(ref_t1w)).structarr
        lta.to_filename(out_lta)

    monkeypatch.setattr(prealign, 'register_session', _fake_register)
    prealign.main([str(bids_dir), str(out_dir), '--participant-label', '01'])

    assert (out_dir / 'dataset_description.json').exists()
    for in_file in sorted(sub.rglob('*.nii.gz')):
        rel = in_file.relative_to(bids_dir)
        expected = REF_AFFINE if rel.parts[1] == 'ses-2' else nb.load(in_file).affine
        assert np.allclose(nb.load(out_dir / rel).affine, expected)
