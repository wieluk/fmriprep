import nibabel as nb
import numpy as np
import pytest
from nitransforms.io.lta import FSLinearTransformArray, VolumeGeometry

from ..registration import _retarget_lta, init_bbreg_wf, init_fsl_bbr_wf


def _init_wf(init_wf, **kwargs):
    params = {'use_bbr': None, 'bold2anat_dof': 6, 'bold2anat_init': 't1w', 'omp_nthreads': 1}
    return init_wf(**{**params, **kwargs})


def _edges(wf, src, dst):
    return wf._graph.get_edge_data(wf.get_node(src), wf.get_node(dst))['connect']


@pytest.mark.parametrize('init_wf', [init_bbreg_wf, init_fsl_bbr_wf])
def test_mri_coreg_args(init_wf):
    assert not _init_wf(init_wf).get_node('mri_coreg').inputs.args
    wf = _init_wf(init_wf, mri_coreg_args='--no-cras0 --no-bf')
    assert wf.get_node('mri_coreg').inputs.args == '--no-cras0 --no-bf'


def test_bbregister_args():
    wf = _init_wf(init_bbreg_wf, bbregister_args='--lh-only')
    assert wf.get_node('bbregister').inputs.args == '--lh-only'


@pytest.mark.parametrize('init_wf', [init_bbreg_wf, init_fsl_bbr_wf])
def test_intermediate(init_wf):
    wf = _init_wf(init_wf, bold2anat_dof=9)
    assert wf.get_node('retarget_lta') is None
    assert wf.get_node('mri_coreg').inputs.dof == 9

    wf = _init_wf(init_wf, bold2anat_dof=9, use_intermediate=True)
    assert ('intermediate', 'source_file') in _edges(wf, 'inputnode', 'mri_coreg')
    assert ('in_file', 'source_file') in _edges(wf, 'inputnode', 'retarget_lta')
    assert wf.get_node('mri_coreg').inputs.dof == 6
    # The BBR fallback is kept
    assert wf.get_node('compare_transforms') is not None


def test_intermediate_header_incompatible():
    with pytest.raises(ValueError, match='header registration'):
        _init_wf(init_bbreg_wf, use_intermediate=True, bold2anat_init='header')


@pytest.mark.parametrize('lta_type', [0, 1])
def test_retarget_lta(tmp_path, monkeypatch, lta_type):
    monkeypatch.chdir(tmp_path)

    def _img(fname, shape, affine):
        nb.Nifti1Image(np.zeros(shape, dtype='u1'), affine).to_filename(tmp_path / fname)
        return str(tmp_path / fname)

    intermediate = _img('int.nii.gz', (40, 48, 44), np.eye(4))
    bold = _img(
        'bold.nii.gz', (20, 20, 8), [[0, -2, 0, 15], [2, 0, 0, 3], [0, 0, 3, 10], [0, 0, 0, 1]]
    )
    anat = _img('anat.nii.gz', (32, 32, 32), np.diag([1.5, 1.5, 1.5, 1]))

    c, s = np.cos(0.3), np.sin(0.3)
    ras2ras = np.array([[1, 0, 0, 4], [0, c, -s, -7], [0, s, c, 2.5], [0, 0, 0, 1]])
    src = VolumeGeometry.from_image(nb.load(intermediate))
    dst = VolumeGeometry.from_image(nb.load(anat))
    lta = FSLinearTransformArray.from_ras(np.linalg.inv(ras2ras)[np.newaxis])
    xfm = lta['xforms'][0]
    xfm.structarr['src'] = src.structarr
    xfm.structarr['dst'] = dst.structarr
    if lta_type == 0:
        xfm.structarr['m_L'] = np.linalg.inv(dst.as_affine()) @ ras2ras @ src.as_affine()
        xfm.structarr['type'] = lta.structarr['type'] = 0
    lta.to_filename(tmp_path / 'in.lta')

    out_xfm = FSLinearTransformArray.from_filename(_retarget_lta('in.lta', bold))['xforms'][0]
    assert np.allclose(out_xfm['m_L'], ras2ras)
    assert np.allclose(out_xfm['src'].as_affine(), nb.load(bold).affine)
