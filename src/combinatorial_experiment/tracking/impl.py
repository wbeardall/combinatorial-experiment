import os
import sqlite3
import warnings
from enum import Enum
from typing import Literal, Optional, Union

default_db_path = os.path.join(".tracking", "jobs.db")

job_id_key = "JOB_ID"
job_tracking_db_key = "JOB_TRACKING_DB"


def set_job_tracking_db_path(path: str):
    os.environ[job_tracking_db_key] = path


def get_job_tracking_db_path(path: Optional[str] = None) -> str:
    if path is None:
        path = os.environ.get(
            job_tracking_db_key, os.path.join(os.path.expanduser("~"), default_db_path)
        )
    return path


class JobTrackingConnection:
    conn: Union[sqlite3.Connection, None] = None

    def get(self) -> sqlite3.Connection:
        if self.conn is None:
            self.conn = sqlite3.connect(get_job_tracking_db_path())
            self.conn.row_factory = sqlite3.Row
        return self.conn


default_tracking_connection = JobTrackingConnection()


def update_job_state(
    *,
    state: Union[str, Enum],
    job_id: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None,
    on_fail: Literal["raise", "warn", "ignore"] = "raise",
):
    try:
        if conn is None:
            conn = default_tracking_connection.get()
        if job_id is None:
            job_id = os.environ.get(job_id_key)
        if job_id is None:
            raise RuntimeError(
                f"job_id not provided, and the {job_id_key} environment variable is not set."
            )
        if isinstance(state, Enum):
            state = state.value
        conn.execute(
            "UPDATE jobs SET state = ?, modified_time = CURRENT_TIMESTAMP WHERE id = ?",
            (state, job_id),
        )
        conn.commit()
    except Exception as e:
        if on_fail == "raise":
            raise RuntimeError("Failed to update job state") from e
        elif on_fail == "warn":
            warnings.warn(f"Failed to update job state: {e}")
