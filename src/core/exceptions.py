class AutoMLOpsError(Exception):
    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={self.message!r}, details={self.details!r})"

class ModelNotFoundError(AutoMLOpsError):
    """Raised when no model with the requested stage/name exists in MLflow."""

class ModelLoadError(AutoMLOpsError):
    """Raised when a model artifact cannot be deserialized or is corrupt."""

class ModelRegistrationError(AutoMLOpsError):
    """Raised when registering or tagging a model in MLflow fails."""

class ModelPromotionError(AutoMLOpsError):
    """Raised when promoting a model from Staging → Production fails."""

class ModelRollbackError(AutoMLOpsError):
    """Raised when rolling back to a previous Production model fails."""

class DataValidationError(AutoMLOpsError):
    """Raised when incoming feature data fails schema or range validation."""

class DataLoadError(AutoMLOpsError):
    """Raised when loading training or reference data fails."""

class FeatureStoreError(AutoMLOpsError):
    """Raised when reading/writing features to the feature store fails."""

class DriftDetectionError(AutoMLOpsError):
    """Raised when the Evidently drift report cannot be generated."""

class InsufficientDataError(AutoMLOpsError):
    """Raised when fewer rows than `drift_window_size` are available for drift check."""

class PipelineTriggerError(AutoMLOpsError):
    """Raised when the Airflow REST API call to trigger a DAG run fails."""

class ValidationGateError(AutoMLOpsError):
    """Raised when the validation gate cannot be evaluated (missing metrics, etc.)."""

class TrainingError(AutoMLOpsError):
    """Raised when model training fails mid-run."""

class DatabaseError(AutoMLOpsError):
    """Raised when a database operation fails after retries."""

class ConfigurationError(AutoMLOpsError):
    """Raised when required configuration is missing or invalid at startup."""
