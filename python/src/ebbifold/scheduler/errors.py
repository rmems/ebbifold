"""Fail-closed errors for the Phase 3a scheduler core."""


class UnsupportedSchedulerInput(ValueError):
    """Raised when a snapshot enables a lane the pure core does not port.

    The TypeScript scheduler has lanes (exam-target treatment, simulation
    deadlines, class-prep blends) whose behavior Phase 3a does not reproduce.
    Rather than silently skipping them, the Python core refuses the input.
    """


class SchedulerDeterminismError(ValueError):
    """Raised when a request would need unseeded randomness to proceed."""
