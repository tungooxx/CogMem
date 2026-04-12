from types import SimpleNamespace

from cogmem.patches.patch import CognitivePatch


class DummyTokenizer:
    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        truncation=False,
        max_length=None,
        padding=False,
        return_dict=False,
        return_assistant_tokens_mask=False,
        **kwargs,
    ):
        if tokenize:
            result = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
            if return_assistant_tokens_mask:
                result["assistant_masks"] = [0, 1, 1]
            return result
        return "chat text"

    def __call__(self, text, truncation=True, max_length=1024, padding=False):
        return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}


class DummyPeftModel:
    def __init__(self, unload_raises=False, disable_raises=False, delete_raises=False):
        self.gradient_checkpointing_disabled = False
        self.adapter_disabled = False
        self.deleted_adapter = None
        self.unload_raises = unload_raises
        self.disable_raises = disable_raises
        self.delete_raises = delete_raises
        self.base_model = SimpleNamespace()
        self.base_model.unloaded = False
        self.base_model.unload = self._unload_base_model

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing_disabled = True

    def disable_adapter_layers(self):
        if self.disable_raises:
            raise RuntimeError("disable failed")
        self.adapter_disabled = True

    def delete_adapter(self, name):
        if self.delete_raises:
            raise RuntimeError("delete failed")
        self.deleted_adapter = name

    def _unload_base_model(self):
        if self.unload_raises:
            raise RuntimeError("unload failed")
        self.base_model.unloaded = True
        return self.base_model


def test_extract_loss_history_monotonic_steps():
    from cogmem.patches.create import _extract_loss_history

    log_history = [
        {"loss": 4.2, "epoch": 1.0, "step": 1},
        {"loss": 3.8, "epoch": 2.0, "step": 2},
        {"train_runtime": 1.23},
        {"loss": 3.7, "epoch": 2.0, "step": 2},
        {"loss": 3.1, "epoch": 3.0, "step": 3},
    ]

    history = _extract_loss_history(log_history)

    assert [entry["step"] for entry in history] == [1, 2, 3]
    assert history[0]["epoch"] == 1.0
    assert history[-1]["loss"] == 3.1


def test_extract_final_loss_prefers_last_loss_then_train_loss():
    from cogmem.patches.create import _extract_final_loss

    assert _extract_final_loss([{"loss": 2.5}, {"train_loss": 9.0}]) == 2.5
    assert _extract_final_loss([{"train_loss": 1.25}]) == 1.25
    assert _extract_final_loss([]) is None


def test_tokenize_training_messages_masks_non_assistant_tokens():
    from cogmem.patches.create import _tokenize_training_messages

    tok = _tokenize_training_messages(
        DummyTokenizer(),
        [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
        max_length=1024,
    )

    assert tok["labels"] == [-100, 2, 3]


def test_tokenize_training_messages_raises_when_assistant_tokens_missing():
    from cogmem.patches.create import _tokenize_training_messages

    class ZeroAssistantTokenizer(DummyTokenizer):
        def apply_chat_template(self, *args, **kwargs):
            if kwargs.get("tokenize"):
                return {
                    "input_ids": [1, 2, 3],
                    "attention_mask": [1, 1, 1],
                    "assistant_masks": [0, 0, 0],
                }
            return super().apply_chat_template(*args, **kwargs)

    try:
        _tokenize_training_messages(
            ZeroAssistantTokenizer(),
            [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
            max_length=1024,
        )
    except ValueError as exc:
        assert "no assistant tokens" in str(exc)
    else:
        raise AssertionError("Expected missing assistant tokens to raise ValueError")


def test_patch_training_callback_collects_logs(capsys):
    from cogmem.patches.create import _PatchTrainingTraceCallback

    callback = _PatchTrainingTraceCallback(total_steps=3, show_progress=True)
    state = SimpleNamespace(global_step=1, epoch=1.0)

    callback.on_log(None, state, None, logs={"loss": 4.0})
    state.global_step = 2
    state.epoch = 2.0
    callback.on_log(None, state, None, logs={"loss": 3.5})

    assert callback.loss_history == [
        {"step": 1, "epoch": 1.0, "loss": 4.0},
        {"step": 2, "epoch": 2.0, "loss": 3.5},
    ]
    assert "step 1/3 | epoch=1.00 | loss=4.0000" in capsys.readouterr().out


def test_create_patch_from_contrast_return_stats_false(monkeypatch):
    from cogmem.patches import create as create_mod

    dummy_model = DummyPeftModel()
    expected_stats = create_mod.PatchTrainingStats(
        total_steps=3,
        final_loss=1.5,
        loss_history=[{"step": 1, "epoch": 1.0, "loss": 2.0}],
    )

    monkeypatch.setattr(create_mod, "get_peft_model", lambda base_model, config: dummy_model)
    monkeypatch.setattr(create_mod, "_train_patch_adapter", lambda **kwargs: expected_stats)
    monkeypatch.setattr(
        create_mod,
        "_extract_lora_weights",
        lambda model: {"layer.weight": {"A": "a", "B": "b"}},
    )
    monkeypatch.setattr(create_mod.torch.cuda, "empty_cache", lambda: None)

    patch = create_mod.create_patch_from_contrast(
        base_model=object(),
        tokenizer=DummyTokenizer(),
        task_prompt="write a function",
        failed_code="return 0",
        passed_code="return 1",
        patch_id="patch_1",
    )

    assert isinstance(patch, CognitivePatch)
    assert patch.patch_id == "patch_1"
    assert patch.rank == create_mod.DEFAULT_PATCH_RANK
    assert patch.lora_weights["layer.weight"]["A"] == "a"
    assert dummy_model.gradient_checkpointing_disabled is True
    assert dummy_model.base_model.unloaded is True


def test_create_patch_from_contrast_return_stats_true(monkeypatch):
    from cogmem.patches import create as create_mod

    dummy_model = DummyPeftModel()
    expected_stats = create_mod.PatchTrainingStats(
        total_steps=5,
        final_loss=0.9,
        loss_history=[
            {"step": 1, "epoch": 1.0, "loss": 1.8},
            {"step": 2, "epoch": 2.0, "loss": 0.9},
        ],
    )
    observed = {}

    def fake_train_patch_adapter(**kwargs):
        observed.update(kwargs)
        return expected_stats

    monkeypatch.setattr(create_mod, "get_peft_model", lambda base_model, config: dummy_model)
    monkeypatch.setattr(create_mod, "_train_patch_adapter", fake_train_patch_adapter)
    monkeypatch.setattr(
        create_mod,
        "_extract_lora_weights",
        lambda model: {"layer.weight": {"A": "a", "B": "b"}},
    )
    monkeypatch.setattr(create_mod.torch.cuda, "empty_cache", lambda: None)

    patch, stats = create_mod.create_patch_from_contrast(
        base_model=object(),
        tokenizer=DummyTokenizer(),
        task_prompt="write a function",
        failed_code="return 0",
        passed_code="return 1",
        patch_id="patch_2",
        n_steps=5,
        lr=1e-3,
        show_progress=True,
        log_every_steps=1,
        return_stats=True,
    )

    assert isinstance(patch, CognitivePatch)
    assert stats == expected_stats
    assert observed["n_steps"] == 5
    assert observed["lr"] == 1e-3
    assert observed["show_progress"] is True
    assert observed["log_every_steps"] == 1
    assert dummy_model.base_model.unloaded is True


def test_ensure_clean_base_model_rejects_stale_peft_layers():
    from cogmem.patches.create import _ensure_clean_base_model
    from peft.tuners.tuners_utils import BaseTunerLayer

    class StaleLayer(BaseTunerLayer):
        def __init__(self):
            self._base_layer = None
            self._disable_adapters = False
            self.merged_adapters = []

        def get_base_layer(self):
            return None

        def merge(self, *args, **kwargs):
            return None

        def unmerge(self, *args, **kwargs):
            return None

    stale_model = SimpleNamespace(modules=lambda: [StaleLayer()])

    try:
        _ensure_clean_base_model(stale_model)
    except RuntimeError as exc:
        assert "already contains PEFT layers" in str(exc)
    else:
        raise AssertionError("Expected stale PEFT layers to raise RuntimeError")


def test_cleanup_patch_training_falls_back_when_unload_fails(monkeypatch):
    from cogmem.patches.create import _cleanup_patch_training

    dummy_model = DummyPeftModel(unload_raises=True)
    monkeypatch.setattr("cogmem.patches.create.gc.collect", lambda: None)
    monkeypatch.setattr("cogmem.patches.create.torch.cuda.empty_cache", lambda: None)

    try:
        _cleanup_patch_training(dummy_model, "does-not-exist")
    except RuntimeError as exc:
        assert "base_model.unload" in str(exc)
    else:
        raise AssertionError("Expected unload failure to be visible")

    assert dummy_model.adapter_disabled is True
    assert dummy_model.deleted_adapter == "default"


def test_cleanup_patch_training_attempts_disable_and_delete_independently(monkeypatch):
    from cogmem.patches.create import _cleanup_patch_training

    dummy_model = DummyPeftModel(unload_raises=True, disable_raises=True)
    monkeypatch.setattr("cogmem.patches.create.gc.collect", lambda: None)
    monkeypatch.setattr("cogmem.patches.create.torch.cuda.empty_cache", lambda: None)

    try:
        _cleanup_patch_training(dummy_model, "does-not-exist")
    except RuntimeError as exc:
        assert "base_model.unload" in str(exc)
        assert "disable_adapter_layers" in str(exc)
    else:
        raise AssertionError("Expected cleanup failure to raise RuntimeError")

    assert dummy_model.deleted_adapter == "default"
