import sys
from pathlib import Path

import pytest

# The harness is not an installed package; put its parent on the path so
# `import swmmbench` works from a bare checkout with no build step.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures() -> Path:
    return FIXTURES
