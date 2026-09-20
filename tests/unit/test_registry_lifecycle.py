"""Registry stage transitions against a real (SQLite-backed) MLflow store.

Mock-based tests cannot catch stage-semantics bugs such as `get_latest_versions`
returning only one version per stage, so these run against the real thing.
"""
from __future__ import annotations

import pytest
from mlflow.tracking import MlflowClient

from src.core.exceptions import ModelRollbackError
from src.model_registry.registry import (
    STAGE_ARCHIVED,
    STAGE_PRODUCTION,
    TAG_PREVIOUS_PRODUCTION,
    TAG_ROLLED_BACK,
    ModelRegistry,
)

MODEL = "lifecycle_model"


@pytest.fixture()
def registry(tmp_path):
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    client.create_registered_model(MODEL)
    for _ in range(3):
        client.create_model_version(MODEL, source="dummy://model")

    reg = ModelRegistry.__new__(ModelRegistry)
    reg._client = client
    reg._model_name = MODEL
    yield reg

    # Release the SQLite files so pytest can delete tmp_path on Windows.
    try:
        for store in (client._tracking_client.store, client._get_registry_client().store):
            engine = getattr(store, "engine", None)
            if engine is not None:
                engine.dispose()
    except Exception:
        pass


def _stages(reg: ModelRegistry) -> dict[str, str]:
    return {str(v.version): v.current_stage for v in reg._client.search_model_versions(f"name='{MODEL}'")}


def _promote_all(reg: ModelRegistry) -> None:
    for v in ("1", "2", "3"):
        reg.promote_to_production(v)


def test_rollback_restores_previous_production_not_the_demoted_model(registry):
    _promote_all(registry)
    assert _stages(registry) == {"1": STAGE_ARCHIVED, "2": STAGE_ARCHIVED, "3": STAGE_PRODUCTION}

    restored = registry.rollback()

    assert restored.version == "2"
    assert _stages(registry) == {"1": STAGE_ARCHIVED, "2": STAGE_PRODUCTION, "3": STAGE_ARCHIVED}


def test_rollback_demoted_model_is_marked_and_never_reselected(registry):
    _promote_all(registry)
    registry.rollback()  # 3 -> 2
    v3_tags = registry._client.get_model_version(MODEL, "3").tags
    assert v3_tags[TAG_ROLLED_BACK] == "true"

    restored = registry.rollback()  # 2 -> 1, must not bounce back to 3

    assert restored.version == "1"
    assert _stages(registry)["1"] == STAGE_PRODUCTION


def test_promotion_records_previous_production(registry):
    _promote_all(registry)
    tags = registry._client.get_model_version(MODEL, "3").tags
    assert tags[TAG_PREVIOUS_PRODUCTION] == "2"


def test_rollback_falls_back_to_newest_older_archived_when_no_lineage_tag(registry):
    # Transition stages directly so no lineage tags exist (e.g. versions promoted by hand).
    for v in ("1", "2", "3"):
        registry._client.transition_model_version_stage(
            MODEL, v, STAGE_PRODUCTION, archive_existing_versions=True
        )

    restored = registry.rollback()

    assert restored.version == "2"


def test_rollback_to_explicit_version(registry):
    _promote_all(registry)

    restored = registry.rollback(target_version="1")

    assert restored.version == "1"
    assert _stages(registry)["3"] == STAGE_ARCHIVED


def test_explicit_target_that_is_already_production_is_rejected(registry):
    _promote_all(registry)

    with pytest.raises(ModelRollbackError):
        registry.rollback(target_version="3")

    assert _stages(registry)["3"] == STAGE_PRODUCTION


def test_unknown_explicit_target_is_rejected_without_demoting_production(registry):
    _promote_all(registry)

    with pytest.raises(ModelRollbackError):
        registry.rollback(target_version="99")

    assert _stages(registry)["3"] == STAGE_PRODUCTION


def test_rollback_with_nothing_to_restore_leaves_production_untouched(registry):
    registry.promote_to_production("1")

    with pytest.raises(ModelRollbackError):
        registry.rollback()

    assert _stages(registry)["1"] == STAGE_PRODUCTION


def test_repromoting_a_rolled_back_version_makes_it_eligible_again(registry):
    _promote_all(registry)
    registry.rollback()  # 3 rolled back -> Production is 2
    registry.promote_to_production("3")  # deliberately live again

    assert TAG_ROLLED_BACK not in registry._client.get_model_version(MODEL, "3").tags
    assert registry.rollback().version == "2"


def test_get_all_versions_returns_every_version_in_a_stage(registry):
    _promote_all(registry)

    archived = registry.get_all_versions(stage=STAGE_ARCHIVED)

    assert [m.version for m in archived] == ["2", "1"]
