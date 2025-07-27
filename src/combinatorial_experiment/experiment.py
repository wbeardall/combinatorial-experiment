import json
import logging
import sqlite3
from dataclasses import asdict, dataclass
from enum import Enum
from sqlite3 import Connection
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

from .utils import NestedDict, escape_identifier

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


def upsert_query(table_name: str) -> str:
    return " ".join(
        [
            f"INSERT INTO {escape_identifier(table_name)} (id, status, config, metrics, metadata, error) VALUES (:id, :status, :config, :metrics, :metadata, :error)",
            "ON CONFLICT(id) DO UPDATE SET status = EXCLUDED.status, config = EXCLUDED.config, metrics = EXCLUDED.metrics, metadata = EXCLUDED.metadata, error = EXCLUDED.error",
        ]
    )


def to_sqlite_compatible(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return json.dumps(value)
    return value


@dataclass
class Experiment:
    id: str
    status: ExperimentStatus
    config: Dict[str, Any]
    metrics: Dict[str, Any]
    metadata: Dict[str, Any]
    error: Union[str, None] = None

    def __post_init__(self):
        if isinstance(self.metrics, str):
            self.metrics = json.loads(self.metrics)
        if isinstance(self.config, str):
            self.config = json.loads(self.config)
        if isinstance(self.metadata, str):
            self.metadata = json.loads(self.metadata)
        if isinstance(self.status, str):
            self.status = ExperimentStatus(self.status)

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "Experiment":
        return cls(**data)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Experiment":
        return cls(
            id=str(uuid4()),
            status=ExperimentStatus.PENDING,
            config=config,
            metrics={},
            metadata={},
        )

    def as_dict(self) -> Dict[str, Any]:
        v = asdict(self)
        v = {k: to_sqlite_compatible(v) for k, v in v.items()}
        return v

    @classmethod
    def from_conn(cls, *, conn: Connection, table_name: str, id: str) -> "Experiment":
        """
        Get an experiment from a SQLite database.
        """
        cursor = conn.execute(
            f"SELECT id, status, config, metrics, metadata, error FROM {escape_identifier(table_name)} WHERE id = ?",
            (id,),
        )
        data = cursor.fetchone()
        return cls.deserialize(data)

    def upsert(
        self,
        *,
        conn: Union[Connection, None] = None,
        table_name: Union[str, None] = None,
    ) -> None:
        conn.execute(upsert_query(table_name), self.as_dict())
        conn.commit()

    def update(
        self,
        *,
        conn: Union[Connection, None] = None,
        table_name: Union[str, None] = None,
        status: Union[str, None] = None,
        metrics: Union[Dict[str, Any], None] = None,
        error: Union[str, None] = None,
        metadata: Union[Dict[str, Any], None] = None,
    ) -> None:
        conn, table_name = self.get_conn(conn=conn, table_name=table_name)
        update_clauses = []
        do_update = False
        data = {"id": self.id}
        if status is not None:
            self.status = status
            update_clauses.append("status = :status")
            data["status"] = status
            do_update = True
        if metrics is not None:
            self.metrics.update(metrics)
            update_clauses.append("metrics = :metrics")
            data["metrics"] = metrics
            do_update = True
        if error is not None:
            self.error = error
            update_clauses.append("error = :error")
            data["error"] = error
            do_update = True
        if metadata is not None:
            self.metadata.update(metadata)
            update_clauses.append("metadata = :metadata")
            data["metadata"] = metadata
            do_update = True
        if do_update:
            update_clause = ", ".join(update_clauses)
            conn.execute(
                f"UPDATE {escape_identifier(table_name)} SET {update_clause} WHERE id = :id",
                {k: to_sqlite_compatible(v) for k, v in data.items()},
            )
            conn.commit()

    def update_status(
        self,
        *,
        conn: Union[Connection, None] = None,
        table_name: Union[str, None] = None,
        status: str,
    ) -> None:
        conn, table_name = self.get_conn(conn=conn, table_name=table_name)
        self.status = status
        conn.execute(
            f"UPDATE {escape_identifier(table_name)} SET status = :status WHERE id = :id",
            {"status": to_sqlite_compatible(status), "id": self.id},
        )
        conn.commit()

    def update_metrics(
        self,
        *,
        conn: Union[Connection, None] = None,
        table_name: Union[str, None] = None,
        metrics: Dict[str, Any],
    ) -> None:
        conn, table_name = self.get_conn(conn=conn, table_name=table_name)
        self.metrics.update(metrics)
        conn.execute(
            f"UPDATE {escape_identifier(table_name)} SET metrics = :metrics WHERE id = :id",
            {"metrics": to_sqlite_compatible(metrics), "id": self.id},
        )
        conn.commit()

    def update_error(
        self,
        *,
        conn: Union[Connection, None] = None,
        table_name: Union[str, None] = None,
        error: str,
    ) -> None:
        conn, table_name = self.get_conn(conn=conn, table_name=table_name)
        self.error = error
        conn.execute(
            f"UPDATE {escape_identifier(table_name)} SET error = :error WHERE id = :id",
            {"error": to_sqlite_compatible(error), "id": self.id},
        )
        conn.commit()

    def update_metadata(
        self,
        *,
        conn: Union[Connection, None] = None,
        table_name: Union[str, None] = None,
        metadata: Dict[str, Any],
    ) -> None:
        conn, table_name = self.get_conn(conn=conn, table_name=table_name)
        self.metadata.update(metadata)
        conn.execute(
            f"UPDATE {escape_identifier(table_name)} SET metadata = :metadata WHERE id = :id",
            {"metadata": to_sqlite_compatible(metadata), "id": self.id},
        )
        conn.commit()

    @property
    def flattened_config(self) -> Dict[str, Any]:
        return NestedDict.flatten(self.config)

    def as_record(self) -> Dict[str, Any]:
        config = {f"CFG_{k}": v for k, v in NestedDict.flatten(self.config).items()}
        metrics = {
            k if k.startswith("METRIC_") or k.startswith("ENG_") else f"METRIC_{k}": v
            for k, v in self.metrics.items()
        }
        metadata = {f"META_{k}": v for k, v in self.metadata.items()}
        return {**config, **metrics, **metadata}

    def get_conn(
        self,
        *,
        conn: Union[Connection, None] = None,
        table_name: Union[str, None] = None,
    ) -> Tuple[Connection, str]:
        if conn is not None and table_name is not None:
            return conn, table_name
        experiment_set = getattr(self, "experiment_set", None)
        if experiment_set is None:
            raise ValueError(
                "Experiment is not bound to an experiment set. Cannot get connection."
            )
        return experiment_set.conn, experiment_set.table_name

    def bind(self, experiment_set: "ExperimentSet"):
        self.experiment_set = experiment_set


def get_table_names(conn: Connection) -> List[str]:
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    return [row[0] for row in cursor]


class ExperimentSet:
    conn: Connection
    table_name: str
    experiments: Dict[str, Experiment]
    logger: logging.Logger

    def __init__(
        self,
        *,
        conn: Connection,
        table_name: str,
        experiments: Union[List[Experiment], None] = None,
        logger: logging.Logger = logger,
    ):
        self.conn = conn
        self.conn.row_factory = sqlite3.Row
        self.table_name = table_name
        self.experiments = {}
        self.ensure_table(conn, table_name)
        self.logger = logger
        if experiments is not None:
            for experiment in experiments:
                self.experiments[experiment.id] = experiment

    @staticmethod
    def ensure_table(conn: Connection, table_name: str) -> None:
        conn.execute(
            f"CREATE TABLE IF NOT EXISTS {escape_identifier(table_name)} (id TEXT PRIMARY KEY, status TEXT, config JSONB, metrics JSONB, metadata JSONB, error TEXT)"
        )

    @classmethod
    def from_path(cls, path: str, table_name: Optional[str] = None) -> "ExperimentSet":
        conn = sqlite3.connect(path)
        return cls.from_conn(conn=conn, table_name=table_name)

    @classmethod
    def from_conn(
        cls, *, conn: Connection, table_name: Optional[str] = None
    ) -> "ExperimentSet":
        if table_name is None:
            table_names = get_table_names(conn)
            if len(table_names) == 1:
                table_name = table_names[0]
            else:
                raise ValueError(
                    f"Cannot infer table name from database with {len(table_names)}  > 1 tables"
                )
        else:
            cls.ensure_table(conn, table_name)
        cursor = conn.execute(
            f"SELECT id, status, config, metrics, metadata, error FROM {escape_identifier(table_name)}"
        )
        deserialized = [Experiment.deserialize(row) for row in cursor]
        return cls(conn=conn, table_name=table_name, experiments=deserialized)

    def get_by_status(self, status: Union[str, ExperimentStatus]) -> List[Experiment]:
        if isinstance(status, str):
            status = ExperimentStatus(status)
        return [
            experiment
            for experiment in self.experiments.values()
            if experiment.status == status
        ]

    def get_all(self) -> List[Experiment]:
        return list(self.experiments.values())

    def get_incomplete(self) -> List[Experiment]:
        return [
            experiment
            for experiment in self.experiments.values()
            if experiment.status != ExperimentStatus.COMPLETED
        ]

    @property
    def complete(self):
        return len(self.get_incomplete()) == 0

    def get_failed(self) -> List[Experiment]:
        return self.get_by_status(ExperimentStatus.FAILED)

    def pull(self) -> None:
        cursor = self.conn.execute(
            f"SELECT id, status, config, metrics, metadata, error FROM {escape_identifier(self.table_name)}"
        )
        count = 0
        for row in cursor:
            deserialized = Experiment.deserialize(row)
            deserialized.bind(self)
            self.experiments[deserialized.id] = deserialized
            count += 1
        self.logger.info(f"Pulled {count} experiments from '{self.table_name}'")

    def push(self, experiments: List[Experiment]) -> None:
        self.conn.executemany(
            upsert_query(self.table_name),
            [experiment.as_dict() for experiment in experiments],
        )
        self.conn.commit()

    def push_all(self) -> None:
        self.push(list(self.experiments.values()))

    def update_experiments(
        self, configs: List[Dict[str, Any]], repeats: int = 1
    ) -> int:
        new_experiments = []
        self.logger.info(
            f"Updating experiments with {len(configs)} configs and {repeats} repeats"
        )
        for config in configs:
            matching = sum(
                1
                for experiment in self.experiments.values()
                if experiment.config == config
            )
            for _ in range(repeats - matching):
                experiment = Experiment.from_config(config)
                experiment.bind(self)
                self.experiments[experiment.id] = experiment
                new_experiments.append(experiment)
        self.logger.info(f"Pushing {len(new_experiments)} new experiments")
        self.push(new_experiments)
        return len(new_experiments)
