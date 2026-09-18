"""Load and validate one site's inputs for the first computation round."""

import uuid

from .types import ValidatedInputs
from .validation import validate_and_get_inputs


def load_inputs(data_dir, parameters, logger):
    """Validate local inputs and build the payload for the first local step.

    Also mints a random ``self_token``: the framework does not expose this
    site's own identity to author code, so the token is how this site later
    recognizes which entry in the aggregator's site-keyed results is its own.
    See local_math.py's ``compute_dimensional_scores``.
    """
    is_valid, data = validate_and_get_inputs(data_dir, parameters, logger)
    if not is_valid:
        raise ValueError("Invalid run input; see the site log for validation details")

    return ValidatedInputs(data=data, self_token=uuid.uuid4().hex)
