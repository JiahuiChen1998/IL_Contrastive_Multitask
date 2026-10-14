"""Filter one repetitive RDKit diagnostic without changing molecular parsing."""
import logging
from rdkit import rdBase


class _IsolatedHydrogenFilter(logging.Filter):
    def filter(self, record):
        return 'WARNING: not removing hydrogen atom without neighbors' not in record.getMessage()


def suppress_isolated_hydrogen_warning():
    logger = logging.getLogger('rdkit')
    if not any(isinstance(f, _IsolatedHydrogenFilter) for f in logger.filters):
        rdBase.LogToPythonLogger()
        logger.addFilter(_IsolatedHydrogenFilter())
