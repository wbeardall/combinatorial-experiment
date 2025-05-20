from .impl import update_job_state

try:
    from schedtools.tracking import update_job_state  # type: ignore
except ImportError:
    pass


__all__ = ["update_job_state"]