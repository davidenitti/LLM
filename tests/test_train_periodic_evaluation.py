from types import SimpleNamespace

import train


class _RecordingArcAGI:
    calls = []

    def __init__(self, data, padding, aligned_labels, **kwargs):
        self.data = data
        self.padding = padding
        self.aligned_labels = aligned_labels
        self.kwargs = kwargs
        _RecordingArcAGI.calls.append(
            {
                "data": data,
                "padding": padding,
                "aligned_labels": aligned_labels,
                "kwargs": kwargs,
            }
        )

    def get_tokenizer(self):
        return object()


class FakeAccelerator:
    def __init__(self, *, is_main_process=True):
        self.is_main_process = is_main_process
        self.wait_calls = 0
        self.logged = []

    def wait_for_everyone(self):
        self.wait_calls += 1

    def unwrap_model(self, model):
        return model

    def log(self, payload, step, log_kwargs):
        self.logged.append((payload, step, log_kwargs))


class FakeTrainMetrics:
    def __init__(self, total_loss_avg):
        self._total_loss_avg = total_loss_avg
        self.reset_total_losses_calls = 0

    def total_loss_avg(self):
        return self._total_loss_avg

    def reset_total_losses(self):
        self.reset_total_losses_calls += 1


class FakeLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(message)


def test_resolve_detach_schedule_without_detach():
    detach_size, mini_steps = train.resolve_detach_schedule(
        sequence_length=128,
        detach_every=None,
        completed_steps=0,
    )

    assert detach_size == 128
    assert mini_steps == 1


def test_resolve_detach_schedule_uses_largest_valid_divisor():
    detach_size, mini_steps = train.resolve_detach_schedule(
        sequence_length=18,
        detach_every=5,
        completed_steps=2000,
    )

    assert detach_size == 3
    assert mini_steps == 6


def test_load_training_datasets_arcagi_keeps_train_stats_and_skips_mask_input_by_default(monkeypatch):
    import arc_agi_dataset

    _RecordingArcAGI.calls = []
    monkeypatch.setattr(arc_agi_dataset, "ArcAGI", _RecordingArcAGI)
    monkeypatch.setattr(
        arc_agi_dataset,
        "get_arc_agi_dataset",
        lambda *args, **kwargs: {"train": ["train_task"], "val": ["val_task"]},
    )

    args = SimpleNamespace(
        dataset_name="custom:arc_agi",
        config_dataset={"aug": True, "gt_labels_only": True},
        debug=False,
        use_generate=False,
        block_size=256,
        aligned_labels=True,
    )
    config = SimpleNamespace(use_pos_emb_2d=False, use_rot_emb_2d=False)

    _, lm_datasets, tokenizer = train.load_training_datasets(args, config=config)

    assert tokenizer is not None
    assert set(lm_datasets) == {"train", "validation", "train_test", "validation_test"}
    assert len(_RecordingArcAGI.calls) == 4

    train_call = _RecordingArcAGI.calls[0]["kwargs"]
    val_call = _RecordingArcAGI.calls[1]["kwargs"]
    train_tta_call = _RecordingArcAGI.calls[2]["kwargs"]
    val_tta_call = _RecordingArcAGI.calls[3]["kwargs"]

    assert train_call["compute_mask_input"] is False
    assert train_call["include_position_idx"] is False

    assert val_call["compute_mask_input"] is False
    assert val_call["include_position_idx"] is False

    assert train_tta_call["compute_mask_input"] is True
    assert val_tta_call["compute_mask_input"] is True


def test_load_training_datasets_arcagi_enables_mask_input_for_generate_accuracy(monkeypatch):
    import arc_agi_dataset

    _RecordingArcAGI.calls = []
    monkeypatch.setattr(arc_agi_dataset, "ArcAGI", _RecordingArcAGI)
    monkeypatch.setattr(
        arc_agi_dataset,
        "get_arc_agi_dataset",
        lambda *args, **kwargs: {"train": ["train_task"], "val": ["val_task"]},
    )

    args = SimpleNamespace(
        dataset_name="custom:arc_agi",
        config_dataset={"aug": True, "gt_labels_only": True},
        debug=False,
        use_generate=True,
        block_size=256,
        aligned_labels=True,
    )
    config = SimpleNamespace(use_pos_emb_2d=False, use_rot_emb_2d=False)

    train.load_training_datasets(args, config=config)

    train_call = _RecordingArcAGI.calls[0]["kwargs"]
    val_call = _RecordingArcAGI.calls[1]["kwargs"]

    assert train_call["compute_mask_input"] is True
    assert val_call["compute_mask_input"] is True


def test_load_training_datasets_arcagi_overrides_conflicting_internal_field_requirements(monkeypatch):
    import arc_agi_dataset

    _RecordingArcAGI.calls = []
    monkeypatch.setattr(arc_agi_dataset, "ArcAGI", _RecordingArcAGI)
    monkeypatch.setattr(
        arc_agi_dataset,
        "get_arc_agi_dataset",
        lambda *args, **kwargs: {"train": ["train_task"], "val": ["val_task"]},
    )

    args = SimpleNamespace(
        dataset_name="custom:arc_agi",
        config_dataset={
            "aug": True,
            "gt_labels_only": True,
        },
        debug=False,
        use_generate=True,
        block_size=256,
        aligned_labels=True,
    )
    config = SimpleNamespace(use_pos_emb_2d=True, use_rot_emb_2d=False)

    train.load_training_datasets(args, config=config)

    train_call = _RecordingArcAGI.calls[0]["kwargs"]
    val_call = _RecordingArcAGI.calls[1]["kwargs"]

    assert train_call["compute_mask_input"] is True
    assert train_call["include_position_idx"] is True
    assert val_call["compute_mask_input"] is True
    assert val_call["include_position_idx"] is True


def test_run_periodic_evaluation_with_stride_and_tracking(monkeypatch):
    calls = {
        "compute_valv2": [],
        "compute_dataset_accuracy": [],
        "sample_generations": [],
    }

    def fake_compute_valv2(
        model,
        dataloader,
        accelerator,
        detach_every,
        generalized,
        precision,
        aligned_labels,
        train_progress,
    ):
        calls["compute_valv2"].append(
            {
                "dataloader": dataloader,
                "detach_every": detach_every,
                "generalized": generalized,
                "precision": precision,
                "aligned_labels": aligned_labels,
                "train_progress": train_progress,
            }
        )
        if dataloader == "eval":
            return 1.5, 4.5, 1000, {"token_acc": 0.7}
        assert dataloader == "stride"
        return 1.75, 5.75, 500, {"token_acc": 0.65}

    def fake_compute_dataset_accuracy(
        args,
        unwrapped_model,
        train_dataloader_single,
        eval_dataloader_single,
        tokenizer,
        completed_steps,
        train_test_dataset,
        validation_test_dataset,
        tta_batch_size_val,
    ):
        calls["compute_dataset_accuracy"].append(
            {
                "completed_steps": completed_steps,
                "tta_batch_size_val": tta_batch_size_val,
                "train_test_dataset": train_test_dataset,
                "validation_test_dataset": validation_test_dataset,
            }
        )
        return {"acc_val": 0.8, "acc_train": 0.9}

    def fake_sample_generations(dataset_name, model, tokenizer, precision, train_progress, steps2think):
        calls["sample_generations"].append(
            {
                "dataset_name": dataset_name,
                "precision": precision,
                "train_progress": train_progress,
                "steps2think": steps2think,
            }
        )

    fake_logger = FakeLogger()
    monkeypatch.setattr(train, "compute_valv2", fake_compute_valv2)
    monkeypatch.setattr(train, "compute_dataset_accuracy", fake_compute_dataset_accuracy)
    monkeypatch.setattr(train, "sample_generations", fake_sample_generations)
    monkeypatch.setattr(train, "logger", fake_logger)

    args = SimpleNamespace(
        dataset_name="custom:arc_agi",
        max_train_steps=200,
        detach_every=16,
        precision="bf16",
        aligned_labels=True,
        per_device_eval_batch_size=8,
        steps2think=3,
        gradient_accumulation_steps=2,
        with_tracking=True,
        total_batch_size=32,
    )
    accelerator = FakeAccelerator(is_main_process=True)
    train_metrics = FakeTrainMetrics(total_loss_avg=0.25)

    perplexity, acc_val_return = train.run_evaluation(
        args=args,
        model=object(),
        eval_dataloader="eval",
        eval_dataloader_stride="stride",
        accelerator=accelerator,
        completed_steps=50,
        train_dataloader_single="train_single",
        eval_dataloader_single="eval_single",
        tokenizer=object(),
        train_test_dataset="train_test",
        validation_test_dataset="validation_test",
        train_metrics=train_metrics,
        num_update_steps_per_epoch=25,
    )

    assert perplexity == 4.5
    assert acc_val_return == 0.8

    assert accelerator.wait_calls == 2
    assert len(calls["compute_valv2"]) == 2
    assert calls["compute_valv2"][0]["dataloader"] == "eval"
    assert calls["compute_valv2"][1]["dataloader"] == "stride"
    assert all(call["generalized"] is True for call in calls["compute_valv2"])
    assert all(call["train_progress"] == 0.25 for call in calls["compute_valv2"])

    assert calls["compute_dataset_accuracy"] == [
        {
            "completed_steps": 50,
            "tta_batch_size_val": 8,
            "train_test_dataset": "train_test",
            "validation_test_dataset": "validation_test",
        }
    ]
    assert calls["sample_generations"] == [
        {
            "dataset_name": "custom:arc_agi",
            "precision": "bf16",
            "train_progress": 0.25,
            "steps2think": 3,
        }
    ]
    assert any("perplexity: 4.5" in message for message in fake_logger.messages)
    assert any("perplexity_stride: 5.75" in message for message in fake_logger.messages)

    assert len(accelerator.logged) == 1
    payload, step, log_kwargs = accelerator.logged[0]
    assert step == 51
    assert log_kwargs == {"wandb": {"commit": True}}
    assert payload["perplexity"] == 4.5
    assert payload["eval_loss"] == 1.5
    assert payload["perplexity_stride"] == 5.75
    assert payload["eval_loss_stride"] == 1.75
    assert payload["train_loss"] == 0.25
    assert payload["epochs"] == 2.0
    assert payload["total_examples"] == 1600
    assert payload["eval_token_acc"] == 0.7
    assert payload["eval_stride_token_acc"] == 0.65
    assert payload["acc_val"] == 0.8
    assert payload["acc_train"] == 0.9
    assert payload["eval_loss_sum"] == 0.0015

    assert train_metrics.reset_total_losses_calls == 1


def test_run_periodic_evaluation_non_main_math_dataset(monkeypatch):
    calls = {
        "compute_valv2": [],
        "compute_dataset_accuracy": 0,
        "sample_generations": 0,
    }

    def fake_compute_valv2(
        model,
        dataloader,
        accelerator,
        detach_every,
        generalized,
        precision,
        aligned_labels,
        train_progress,
    ):
        calls["compute_valv2"].append(
            {
                "generalized": generalized,
                "train_progress": train_progress,
            }
        )
        return 2.0, 6.0, None, {}

    def fake_compute_dataset_accuracy(*args, **kwargs):
        calls["compute_dataset_accuracy"] += 1
        raise AssertionError("compute_dataset_accuracy should not run on non-main processes")

    def fake_sample_generations(*args, **kwargs):
        calls["sample_generations"] += 1
        raise AssertionError("sample_generations should not run on non-main processes")

    fake_logger = FakeLogger()
    monkeypatch.setattr(train, "compute_valv2", fake_compute_valv2)
    monkeypatch.setattr(train, "compute_dataset_accuracy", fake_compute_dataset_accuracy)
    monkeypatch.setattr(train, "sample_generations", fake_sample_generations)
    monkeypatch.setattr(train, "logger", fake_logger)

    args = SimpleNamespace(
        dataset_name="custom:math_dataset",
        max_train_steps=100,
        detach_every=None,
        precision="fp32",
        aligned_labels=False,
        per_device_eval_batch_size=4,
        steps2think=1,
        gradient_accumulation_steps=1,
        with_tracking=False,
        total_batch_size=16,
    )
    accelerator = FakeAccelerator(is_main_process=False)
    train_metrics = FakeTrainMetrics(total_loss_avg=0.4)

    perplexity, acc_val_return = train.run_evaluation(
        args=args,
        model=object(),
        eval_dataloader="eval",
        eval_dataloader_stride=None,
        accelerator=accelerator,
        completed_steps=40,
        train_dataloader_single="train_single",
        eval_dataloader_single="eval_single",
        tokenizer=object(),
        train_test_dataset=None,
        validation_test_dataset=None,
        train_metrics=train_metrics,
        num_update_steps_per_epoch=20,
    )

    assert perplexity == 6.0
    assert acc_val_return is None

    assert accelerator.wait_calls == 2
    assert accelerator.logged == []
    assert calls["compute_dataset_accuracy"] == 0
    assert calls["sample_generations"] == 0
    assert calls["compute_valv2"] == [{"generalized": False, "train_progress": 0.4}]
    assert any("Validation perplexity: 6.0" in message for message in fake_logger.messages)
    assert train_metrics.reset_total_losses_calls == 0
