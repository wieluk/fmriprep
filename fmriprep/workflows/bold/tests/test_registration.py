import pytest

from ..registration import init_bbreg_wf, init_fsl_bbr_wf


def _init_wf(init_wf, **kwargs):
    params = {'use_bbr': None, 'bold2anat_dof': 6, 'bold2anat_init': 't1w', 'omp_nthreads': 1}
    return init_wf(**{**params, **kwargs})


@pytest.mark.parametrize('init_wf', [init_bbreg_wf, init_fsl_bbr_wf])
def test_mri_coreg_args(init_wf):
    assert not _init_wf(init_wf).get_node('mri_coreg').inputs.args
    wf = _init_wf(init_wf, mri_coreg_args='--no-cras0 --no-bf')
    assert wf.get_node('mri_coreg').inputs.args == '--no-cras0 --no-bf'


def test_bbregister_args():
    wf = _init_wf(init_bbreg_wf, bbregister_args='--lh-only')
    assert wf.get_node('bbregister').inputs.args == '--lh-only'
