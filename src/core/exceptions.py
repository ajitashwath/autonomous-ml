class AutoMLOpsError(Exception):
    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={self.message!r}, details={self.details!r})"


class ModelNotFoundError(AutoMLOpsError):
    pass


class ModelLoadError(AutoMLOpsError):
    pass


class ModelRegistrationError(AutoMLOpsError):
    pass


class ModelPromotionError(AutoMLOpsError):
    pass


class ModelRollbackError(AutoMLOpsError):
    pass


class DataValidationError(AutoMLOpsError):
    pass


class DataLoadError(AutoMLOpsError):
    pass


class FeatureStoreError(AutoMLOpsError):
    pass


class DriftDetectionError(AutoMLOpsError):
    pass


class InsufficientDataError(AutoMLOpsError):
    pass


class PipelineTriggerError(AutoMLOpsError):
    pass


class ValidationGateError(AutoMLOpsError):
    pass


class TrainingError(AutoMLOpsError):
    pass


class DatabaseError(AutoMLOpsError):
    pass


class ConfigurationError(AutoMLOpsError):
    pass
