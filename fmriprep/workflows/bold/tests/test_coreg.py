from pathlib import Path

import bids
import pytest
from niworkflows.utils.testing import generate_bids_skeleton

from ..coreg import get_coreg_intermediate


def _bold():
    return {'task': 'rest', 'suffix': 'bold', 'metadata': {'RepetitionTime': 2.0}}


@pytest.fixture(scope='module')
def layout(tmp_path_factory):
    bids_dir = tmp_path_factory.mktemp('intermediates') / 'bids'
    generate_bids_skeleton(
        bids_dir,
        {
            '01': [
                {
                    'session': '1',
                    'anat': [{'suffix': 'T1w'}, {'suffix': 'T2w'}],
                    'func': [_bold()],
                },
                {'session': '2', 'anat': [{'suffix': 'T2w'}], 'func': [_bold()]},
                {'session': '3', 'func': [_bold()]},
            ],
            '02': {'anat': [{'suffix': 'T1w'}], 'func': [_bold()]},
        },
    )
    return bids.BIDSLayout(bids_dir)


@pytest.mark.parametrize(
    ('subject', 'session', 'expected'),
    [
        ('01', '1', 'sub-01_ses-1_T1w.nii.gz'),
        ('01', '2', 'sub-01_ses-2_T2w.nii.gz'),
        ('01', '3', None),
        ('02', None, None),
    ],
)
def test_get_coreg_intermediate(layout, subject, session, expected):
    (bold_file,) = layout.get(
        subject=subject, session=session, suffix='bold', extension='.nii.gz', return_type='file'
    )
    intermediate = get_coreg_intermediate(bold_file, layout)
    assert (intermediate and Path(intermediate).name) == expected
