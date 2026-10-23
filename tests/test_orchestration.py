from dagster import Definitions

from pitwall.orchestration.definitions import defs


def test_definitions_load_and_graph_is_connected():
    Definitions.validate_loadable(defs)
    graph = defs.get_repository_def().asset_graph
    keys = {k.to_user_string() for k in graph.get_all_asset_keys()}
    assert {"bronze_jolpica", "silver_results", "gold_dims", "gold_fact_session_result"} <= keys
    job = defs.resolve_job_def("season_refresh")
    assert job.partitions_def is not None
