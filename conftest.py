''' conftest.py '''
import os
import shutil
import pytest


@pytest.fixture(autouse=True)
def reset_const():
    TEST_CHECKPOINT = os.path.join('checkpoints', 'TEST')
    if os.path.isdir(TEST_CHECKPOINT):
        shutil.rmtree(TEST_CHECKPOINT)
