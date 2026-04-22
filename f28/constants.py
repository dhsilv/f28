"""
F28 shared constants. Single source of truth for any scalar that would
otherwise be duplicated across modules.
"""
from __future__ import annotations

# Calendar-time year in seconds. CL trades ~24/5 and its carry model is
# driven by physical storage, which accrues on calendar time (not trading
# time). Every module that converts elapsed seconds to tau MUST use this
# constant so the EKF process-noise and tau inputs are consistent.
SECONDS_PER_YEAR: float = 365.25 * 24.0 * 3600.0
