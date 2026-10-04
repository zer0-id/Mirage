from collections.abc import Generator
from contextlib import contextmanager

from tpm2_pytss import ESAPI
from tpm2_pytss.constants import ESYS_TR

PRIMARY_TEMPLATE = "ecc256"


@contextmanager
def primary(ectx: ESAPI) -> Generator[ESYS_TR, None, None]:
    """Create the primary key and always flush it on exit."""
    handle = ectx.create_primary(None, PRIMARY_TEMPLATE)[0]
    try:
        yield handle
    finally:
        ectx.flush_context(handle)
