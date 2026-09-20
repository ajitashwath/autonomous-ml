from src.core.logging import get_logger
from src.model_registry.registry import ModelRegistry

logger = get_logger(__name__)

def deploy_model(registry: ModelRegistry, version: str, reason: str) -> None:
    logger.info("auto_deployer_starting", version=version, reason=reason)
    registry.promote_to_production(version=version, archive_existing=True)
    registry.annotate_version(
        version=version,
        description=f"[AUTO-DEPLOYED] {reason}"
    )
    logger.info("auto_deployer_finished", version=version)
