from cogmem.consolidation.adapter_registry import (
    AdapterRegistry,
    register_adapter_artifact,
    write_adapter_metadata,
)


def test_registry_roundtrip_and_selection(tmp_path):
    registry_path = tmp_path / "registry.json"
    adapter_a = tmp_path / "global_adapter"
    adapter_b = tmp_path / "file_io_adapter"

    register_adapter_artifact(
        str(registry_path),
        str(adapter_a),
        base_model="Qwen/Qwen2.5-3B-Instruct",
        adapter_role="global",
        training_manifest_ids=["manifest_a"],
        source_skill_card_ids=["skill_global"],
        compatible_families=["general"],
        dev_gain=0.2,
    )
    register_adapter_artifact(
        str(registry_path),
        str(adapter_b),
        base_model="Qwen/Qwen2.5-3B-Instruct",
        adapter_role="family",
        training_manifest_ids=["manifest_a"],
        source_skill_card_ids=["skill_file_io"],
        compatible_families=["file_io"],
        dev_gain=0.4,
    )

    registry = AdapterRegistry.load(str(registry_path))

    assert len(registry) == 2
    assert registry.filter(adapter_role="global")[0].adapter_path == str(adapter_a)
    routed = registry.select_routed_adapters(task_family="file_io", min_dev_gain=0.1)
    assert routed["global"].adapter_path == str(adapter_a)
    assert routed["family"].adapter_path == str(adapter_b)


def test_write_adapter_metadata_creates_lineage_file(tmp_path):
    adapter_dir = tmp_path / "adapter"
    payload = write_adapter_metadata(
        str(adapter_dir),
        base_model="Qwen/Qwen2.5-3B-Instruct",
        adapter_role="family",
        training_manifest_ids=["manifest_a"],
        source_skill_card_ids=["skill_a"],
        compatible_families=["file_io"],
        dev_gain=0.15,
        metadata={"trainer": "unit"},
    )

    assert (adapter_dir / "adapter_metadata.json").exists()
    assert payload["training_manifest_ids"] == ["manifest_a"]
    assert payload["source_skill_card_ids"] == ["skill_a"]
