import datetime
import uuid

from sqlalchemy import JSON, UUID, Boolean, DateTime, Float, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass

class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    request_id: Mapped[str] = mapped_column(
        String(36),
        index=True,
        doc="API request UUID for tracing end-to-end",
    )
    features: Mapped[dict] = mapped_column(
        JSON,
        doc="The raw feature dict received by the API (pre-validation)",
    )

    prediction: Mapped[int] = mapped_column(Integer)
    probability: Mapped[float] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String(50))
    drift_analyzed: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        index=True,
        doc="Flag indicating if this prediction has been processed by the drift detector",
    )

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.datetime.now(datetime.timezone.utc),
        index=True,
    )

    # Ground truth arrives later (POST /api/v1/labels); NULL until then. Rows with a label
    # are what retraining and the validation gate learn from and are scored on.
    actual_label: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        index=True,
        doc="Real outcome (0 = stayed, 1 = churned) once known",
    )
    label_received_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    def __repr__(self) -> str:
        return f"<PredictionLog {self.request_id} -> {self.prediction}>"
