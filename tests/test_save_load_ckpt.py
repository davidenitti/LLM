import os
from pathlib import Path
from types import SimpleNamespace

import torch
from accelerate import Accelerator

from utils import save_load_ckpt


def _write_checkpoint_pair(base_dir, step):
    model_dir = Path(base_dir) / f"step_{step}"
    accelerator_dir = Path(base_dir) / f"step_{step}_accelerator"
    model_dir.mkdir(parents=True, exist_ok=True)
    accelerator_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "pytorch_model.bin").write_bytes(b"weights")
    (accelerator_dir / "model.safetensors").write_bytes(b"state")


class _RecordingAccelerator:
    def __init__(self, *, fail_on_save=False, load_failures=None):
        self.device = torch.device("cpu")
        self.num_processes = 1
        self.is_main_process = True
        self.fail_on_save = fail_on_save
        self.load_failures = {os.fspath(path) for path in (load_failures or [])}
        self.loaded_paths = []
        self.saved_state_paths = []
        self.printed = []
        self.wait_calls = 0

    def wait_for_everyone(self):
        self.wait_calls += 1

    def save_state(self, output_dir):
        output_dir = Path(output_dir)
        self.saved_state_paths.append(os.fspath(output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "model.safetensors").write_bytes(b"state")
        if self.fail_on_save:
            raise RuntimeError("boom")

    def unwrap_model(self, model):
        return model

    def save(self, obj, path):
        torch.save(obj, path)

    def print(self, message):
        self.printed.append(message)

    def load_state(self, checkpoint_path):
        checkpoint_path = os.fspath(checkpoint_path)
        if checkpoint_path in self.load_failures:
            raise RuntimeError("corrupt checkpoint")
        self.loaded_paths.append(checkpoint_path)


class _FakeModel:
    def __init__(self):
        self.saved_paths = []

    def save_pretrained(self, output_dir, is_main_process=True, save_function=None):
        self.saved_paths.append(os.fspath(output_dir))
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "pytorch_model.bin"), "wb") as f:
            f.write(b"weights")
        with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
            f.write("{}")


class _FakeTokenizer:
    def __init__(self):
        self.saved_paths = []

    def save_pretrained(self, output_dir):
        self.saved_paths.append(os.fspath(output_dir))
        with open(os.path.join(output_dir, "tokenizer.json"), "w", encoding="utf-8") as f:
            f.write("{}")
        with open(os.path.join(output_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
            f.write("{}")


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(2, 2)

    def forward(self, x):
        return self.lin(x)

    def save_pretrained(self, output_dir, is_main_process=True, save_function=None):
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
            f.write("{}")


class _TinyTokenizer:
    def save_pretrained(self, output_dir):
        with open(os.path.join(output_dir, "tokenizer.json"), "w", encoding="utf-8") as f:
            f.write("{}")
        with open(os.path.join(output_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
            f.write("{}")


def test_resume_training_loads_previous_checkpoint_if_latest_fails(tmp_path, monkeypatch):
    _write_checkpoint_pair(tmp_path, 5)
    _write_checkpoint_pair(tmp_path, 10)

    bad_checkpoint = str(tmp_path / "step_10_accelerator")
    accelerator = _RecordingAccelerator(load_failures={bad_checkpoint})
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    args = SimpleNamespace(
        resume="",
        output_dir=str(tmp_path),
        reset_resume=False,
        weight_decay=0.0,
        gradient_accumulation_steps=1,
        scale_lowrank=False,
    )
    monkeypatch.setattr(save_load_ckpt, "update_optimizer_weight_decay", lambda *args, **kwargs: None)

    ckpt_path, _, _, starting_epoch, completed_steps, resume_step = save_load_ckpt.resume_training(
        args=args,
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_dataloader=[0, 1, 2],
        num_update_steps_per_epoch=3,
        num_training_steps=9,
        warmup_steps=0,
        scheduler_specific_kwargs={},
    )

    assert ckpt_path == str(tmp_path / "step_5_accelerator")
    assert accelerator.loaded_paths == [str(tmp_path / "step_5_accelerator")]
    assert any("Failed to load checkpoint" in message for message in accelerator.printed)
    assert any("Loaded fallback checkpoint" in message for message in accelerator.printed)
    assert starting_epoch == 2
    assert completed_steps == 5
    assert resume_step == 2


def test_resume_training_uses_explicit_resume_then_falls_back_to_previous(tmp_path, monkeypatch):
    _write_checkpoint_pair(tmp_path, 5)
    _write_checkpoint_pair(tmp_path, 10)

    bad_checkpoint = str(tmp_path / "step_10_accelerator")
    accelerator = _RecordingAccelerator(load_failures={bad_checkpoint})
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    args = SimpleNamespace(
        resume=bad_checkpoint,
        output_dir=str(tmp_path),
        reset_resume=False,
        weight_decay=0.0,
        gradient_accumulation_steps=1,
        scale_lowrank=False,
    )
    monkeypatch.setattr(save_load_ckpt, "update_optimizer_weight_decay", lambda *args, **kwargs: None)

    ckpt_path, _, _, starting_epoch, completed_steps, resume_step = save_load_ckpt.resume_training(
        args=args,
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_dataloader=[0, 1, 2],
        num_update_steps_per_epoch=3,
        num_training_steps=9,
        warmup_steps=0,
        scheduler_specific_kwargs={},
    )

    assert ckpt_path == str(tmp_path / "step_5_accelerator")
    assert accelerator.loaded_paths == [str(tmp_path / "step_5_accelerator")]
    assert accelerator.printed[0] == f"Resume from checkpoint: {bad_checkpoint}"
    assert any("Loaded fallback checkpoint" in message for message in accelerator.printed)
    assert starting_epoch == 2
    assert completed_steps == 5
    assert resume_step == 2


def test_save_checkpoint_failed_save_keeps_previous_checkpoint(tmp_path):
    _write_checkpoint_pair(tmp_path, 1)

    accelerator = _RecordingAccelerator(fail_on_save=True)
    args = SimpleNamespace(output_dir=str(tmp_path))

    save_load_ckpt.save_checkpoint(
        accelerator=accelerator,
        model=_FakeModel(),
        tokenizer=_FakeTokenizer(),
        args=args,
        completed_steps=2,
    )

    assert (tmp_path / "step_1").exists()
    assert (tmp_path / "step_1_accelerator").exists()
    assert not (tmp_path / "step_2").exists()
    assert not (tmp_path / "step_2_accelerator").exists()


def test_save_checkpoint_saves_directly_to_step_dirs(tmp_path):
    accelerator = _RecordingAccelerator()
    model = _FakeModel()
    tokenizer = _FakeTokenizer()
    args = SimpleNamespace(output_dir=str(tmp_path))

    save_load_ckpt.save_checkpoint(
        accelerator=accelerator,
        model=model,
        tokenizer=tokenizer,
        args=args,
        completed_steps=2,
    )

    assert accelerator.saved_state_paths == [str(tmp_path / "step_2_accelerator")]
    assert model.saved_paths == [str(tmp_path / "step_2")]
    assert tokenizer.saved_paths == [str(tmp_path / "step_2")]


def test_save_checkpoint_keeps_two_latest_checkpoint_pairs(tmp_path):
    accelerator = _RecordingAccelerator()
    args = SimpleNamespace(output_dir=str(tmp_path))

    for step in (1, 2, 3):
        save_load_ckpt.save_checkpoint(
            accelerator=accelerator,
            model=_FakeModel(),
            tokenizer=_FakeTokenizer(),
            args=args,
            completed_steps=step,
        )

    remaining = sorted(path.name for path in tmp_path.iterdir())

    assert remaining == ["step_2", "step_2_accelerator", "step_3", "step_3_accelerator"]


def test_save_checkpoint_and_resume_round_trip_with_real_accelerator(tmp_path):
    accelerator = Accelerator(cpu=True)
    model = _TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=0.01)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    x = torch.randn(2, 2)
    loss = model(x).sum()
    accelerator.backward(loss)
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()

    expected_weight = accelerator.unwrap_model(model).lin.weight.detach().clone()

    args = SimpleNamespace(
        output_dir=str(tmp_path),
        resume="",
        reset_resume=False,
        weight_decay=0.01,
        gradient_accumulation_steps=1,
        scale_lowrank=False,
    )

    save_load_ckpt.save_checkpoint(
        accelerator=accelerator,
        model=model,
        tokenizer=_TinyTokenizer(),
        args=args,
        completed_steps=5,
    )

    with torch.no_grad():
        accelerator.unwrap_model(model).lin.weight.zero_()
        accelerator.unwrap_model(model).lin.bias.zero_()

    ckpt_path, _, _, starting_epoch, completed_steps, resume_step = save_load_ckpt.resume_training(
        args=args,
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_dataloader=[0, 1, 2, 3],
        num_update_steps_per_epoch=4,
        num_training_steps=12,
        warmup_steps=0,
        scheduler_specific_kwargs={},
    )

    restored_weight = accelerator.unwrap_model(model).lin.weight.detach()

    assert ckpt_path == str(tmp_path / "step_5_accelerator")
    assert torch.allclose(restored_weight, expected_weight)
    assert starting_epoch == 2
    assert completed_steps == 5
    assert resume_step == 1
