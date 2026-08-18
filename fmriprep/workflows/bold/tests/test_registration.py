import pytest

from ..registration import init_bbreg_wf, init_fsl_bbr_wf


def _bbreg(**kwargs):
    params = {
        'use_bbr': True,
        'bold2anat_dof': 6,
        'bold2anat_init': 't1w',
        'omp_nthreads': 1,
    }
    params.update(kwargs)
    return init_bbreg_wf(**params)


def test_bbreg_intermediate_wiring():
    wf = _bbreg(use_intermediate=True)

    bbregister = wf.get_node('bbregister')
    assert bbregister.inputs.init == 'coreg'
    # bbregister ignores --int when --init-reg is given, so our mri_coreg must not run
    assert wf.get_node('mri_coreg') is None

    edge = wf._graph.get_edge_data(wf.get_node('inputnode'), bbregister)
    assert ('intermediate', 'intermediate_file') in edge['connect']


def test_bbreg_no_intermediate_keeps_coreg_init():
    wf = _bbreg()

    assert wf.get_node('mri_coreg') is not None
    edge = wf._graph.get_edge_data(wf.get_node('inputnode'), wf.get_node('bbregister'))
    assert not any(dest == 'intermediate_file' for _, dest in edge['connect'])


def test_bbreg_intermediate_disables_fallback():
    # use_bbr=None would otherwise compare BBR against mri_coreg, which we no longer run
    wf = _bbreg(use_bbr=None, use_intermediate=True)

    assert wf.get_node('compare_transforms') is None
    assert wf.get_node('outputnode').inputs.fallback is False


@pytest.mark.parametrize(
    ('kwargs', 'match'),
    [
        ({'use_bbr': False}, 'Cannot disable BBR'),
        ({'bold2anat_init': 'header'}, 'header registration'),
    ],
)
def test_bbreg_intermediate_incompatible(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _bbreg(use_intermediate=True, **kwargs)


def test_fsl_bbr_rejects_intermediate():
    with pytest.raises(NotImplementedError, match='not supported for FSL'):
        init_fsl_bbr_wf(
            use_bbr=True,
            bold2anat_dof=6,
            bold2anat_init='t1w',
            omp_nthreads=1,
            use_intermediate=True,
        )
