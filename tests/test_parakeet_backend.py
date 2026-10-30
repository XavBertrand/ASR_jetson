"""Backend isolation, bounded audio decoding and public CLI/export contracts."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from asr_jetson.asr import parakeet_engine as pe
from asr_jetson.pipeline import cli
from asr_jetson.pipeline import full_pipeline as fp


class Recognizer:
    def __init__(self):
        self.sizes = []

    def create_stream(self):
        return SimpleNamespace(
            accept_waveform=lambda rate, samples: self.sizes.append((rate, len(samples))),
            result=SimpleNamespace(text=" Bonjour. "),
        )

    def decode_stream(self, stream):
        pass


def test_bounded_decode_absolute_offsets_overlap_and_clamping(tmp_path):
    audio = tmp_path / "audio.wav"
    sf.write(audio, np.zeros(64000), 16000)
    model = Recognizer()
    result = pe.transcribe_parakeet(
        model,
        audio,
        [
            {"start": 1, "end": 3},
            {"start": 2, "end": 9},
            {"start": -1, "end": 0.5},
            {"start": 2, "end": 1},
        ],
        chunk_seconds=1,
    )
    assert [(s["start"], s["end"]) for s in result] == [
        (0, 0.5),
        (1, 2),
        (2, 3),
        (3, 4),
    ]
    assert all(s["text"] == "Bonjour." for s in result)
    assert max(n for _, n in model.sizes) <= 16000


@pytest.mark.parametrize("rate,channels", [(8000, 1), (16000, 2)])
def test_rejects_wrong_audio(tmp_path, rate, channels):
    path = tmp_path / "audio.wav"
    sf.write(path, np.zeros((100, channels)), rate)
    with pytest.raises(ValueError, match="16 kHz mono"):
        pe.transcribe_parakeet(Recognizer(), path, [])


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf")])
def test_invalid_window(seconds):
    with pytest.raises(ValueError, match="positive"):
        fp.PipelineConfig(parakeet_chunk_seconds=seconds)


def test_missing_runtime_actionable(monkeypatch):
    monkeypatch.setitem(sys.modules, "sherpa_onnx", None)
    with pytest.raises(RuntimeError, match="--extra parakeet"):
        pe.load_parakeet()


def test_local_model_and_runtime_configuration(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "sherpa_onnx",
        SimpleNamespace(
            OfflineRecognizer=SimpleNamespace(from_transducer=lambda **kw: calls.append(kw))
        ),
    )
    with pytest.raises(FileNotFoundError, match="encoder"):
        pe.load_parakeet(tmp_path)
    for name in pe.MODEL_FILES:
        (tmp_path / name).touch()
    pe.load_parakeet(tmp_path)
    assert calls[0]["provider"] == "cpu"
    assert calls[0]["model_type"] == "nemo_transducer"


@pytest.mark.parametrize("backend", ["whisper", "parakeet"])
def test_cli_backend_selection(monkeypatch, tmp_path, backend):
    captured = []
    monkeypatch.setattr(cli, "run_pipeline", lambda audio, cfg: captured.append(cfg) or {})
    argv = ["--audio", "sample.wav"]
    if backend == "parakeet":
        argv += ["--asr-backend", backend, "--parakeet-model-dir", str(tmp_path)]
    cli.main(argv)
    assert captured[0].asr_backend == backend
    assert captured[0].parakeet_model_dir == (tmp_path if backend == "parakeet" else None)


@pytest.mark.parametrize("backend", ["whisper", "parakeet"])
def test_pipeline_exports_and_backend_isolation(monkeypatch, tmp_path, backend):
    import json

    wav = tmp_path / "input.wav"
    sf.write(wav, np.zeros(16000), 16000)
    monkeypatch.setattr(fp, "convert_to_wav", lambda *a, **kw: wav)
    monkeypatch.setattr(
        fp, "apply_diarization", lambda *a, **kw: [{"start": 0, "end": 1, "speaker": 0}]
    )
    monkeypatch.setattr(
        fp, "_run_postprocessing", lambda **kw: ({"txt": str(kw["base_text_path"])}, {})
    )

    def forbidden(*a, **kw):
        pytest.fail("Unselected backend was invoked")

    if backend == "whisper":
        monkeypatch.setattr(fp, "load_faster_whisper", lambda **kw: (object(), {}))
        monkeypatch.setattr(
            fp,
            "transcribe_segments",
            lambda *a, **kw: [{"start": 0.0, "end": 1.0, "text": "Bonjour."}],
        )
        monkeypatch.setattr(pe, "load_parakeet", forbidden)
    else:
        monkeypatch.setattr(fp, "load_faster_whisper", forbidden)
        monkeypatch.setattr(fp, "transcribe_segments", forbidden)
        monkeypatch.setattr(pe, "load_parakeet", lambda *a, **kw: Recognizer())
    cfg = fp.PipelineConfig(out_dir=tmp_path / "out", asr_backend=backend)
    result = fp.run_pipeline(wav, cfg)
    payload = json.loads(Path(result["json"]).read_text())
    assert payload["segments"][0]["speaker"] == 0
    assert payload["segments"][0]["text"] == "Bonjour."
    assert "SPK0" in Path(result["srt"]).read_text()
    manifest = json.loads((cfg.out_dir / "manifest.json").read_text())
    assert manifest["pipeline"]["asr_backend"] == backend
    expected_tag = "medium" if backend == "whisper" else "parakeet-tdt-0.6b-v3"
    assert Path(result["json"]).stem.endswith(expected_tag)


@pytest.mark.integration
def test_real_parakeet_transcription():
    """Opt-in real weights test: ASR_TEST_PARAKEET_MODEL=/path/to/export."""
    import os
    import tempfile

    from asr_jetson.preprocessing.convert_to_wav import convert_to_wav

    root = os.environ.get("ASR_TEST_PARAKEET_MODEL")
    if not root:
        pytest.skip("Set ASR_TEST_PARAKEET_MODEL to a local Parakeet v3 INT8 export")
    with tempfile.TemporaryDirectory() as folder:
        wav = convert_to_wav(
            Path(__file__).parent / "data" / "test.mp3", Path(folder) / "input.wav"
        )
        result = pe.transcribe_parakeet(
            pe.load_parakeet(root, device=os.environ.get("ASR_TEST_PARAKEET_DEVICE", "cpu")),
            wav,
            [{"start": 0, "end": 20}],
        )
    assert result and result[0]["text"]
    assert 0 <= result[0]["start"] < result[0]["end"] <= 20


def test_automatic_download_is_limited_to_runtime_files(monkeypatch, tmp_path):
    import huggingface_hub

    calls = []
    for name in pe.MODEL_FILES:
        (tmp_path / name).touch()
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda repo, **kw: calls.append((repo, kw)) or str(tmp_path),
    )
    monkeypatch.setitem(
        sys.modules,
        "sherpa_onnx",
        SimpleNamespace(OfflineRecognizer=SimpleNamespace(from_transducer=lambda **kw: object())),
    )
    pe.load_parakeet()
    assert calls == [(pe.MODEL_REPO, {"allow_patterns": list(pe.MODEL_FILES)})]


def test_no_speech_or_empty_text_produces_no_segments(tmp_path):
    wav = tmp_path / "silence.wav"
    sf.write(wav, np.zeros(16000), 16000)
    model = Recognizer()
    assert pe.transcribe_parakeet(model, wav, []) == []
    assert model.sizes == []
    model.create_stream = lambda: SimpleNamespace(
        accept_waveform=lambda *a: None, result=SimpleNamespace(text=" ")
    )
    assert pe.transcribe_parakeet(model, wav, [{"start": 0, "end": 1}]) == []


def test_invalid_backend_rejected():
    with pytest.raises(ValueError, match="asr_backend"):
        fp.PipelineConfig(asr_backend="unknown")
    with pytest.raises(SystemExit) as error:
        cli.main(["--audio", "audio.wav", "--asr-backend", "unknown"])
    assert error.value.code == 2


def test_cuda_selects_float_export_and_provider(monkeypatch, tmp_path):
    import huggingface_hub

    calls = []
    checks = []
    downloads = []
    for name in pe.CUDA_MODEL_FILES:
        (tmp_path / name).touch()
    monkeypatch.setitem(
        sys.modules,
        "sherpa_onnx",
        SimpleNamespace(
            OfflineRecognizer=SimpleNamespace(from_transducer=lambda **kw: calls.append(kw))
        ),
    )
    monkeypatch.setattr(pe, "_require_cuda_runtime", lambda runtime: checks.append(runtime))
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda repo, **kw: downloads.append((repo, kw)) or str(tmp_path),
    )
    pe.load_parakeet(device="cuda")
    assert checks
    assert calls[0]["provider"] == "cuda"
    assert calls[0]["encoder"].endswith("encoder.onnx")
    assert downloads == [(pe.CUDA_MODEL_REPO, {"allow_patterns": list(pe.CUDA_MODEL_FILES)})]
    (tmp_path / "encoder.weights").unlink()
    with pytest.raises(FileNotFoundError, match="encoder.weights"):
        pe.load_parakeet(tmp_path, device="cuda")


def test_cuda_rejects_cpu_runtime():
    with pytest.raises(RuntimeError, match="--extra parakeet-cuda"):
        pe._require_cuda_runtime(SimpleNamespace(__version__="2.0.1"))


def test_cuda_rejects_unavailable_gpu(monkeypatch):
    monkeypatch.setattr(fp.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no CUDA GPU"):
        pe._require_cuda_runtime(SimpleNamespace(__version__="2.0.1+cuda"))
    with pytest.raises(RuntimeError, match="no CUDA GPU"):
        fp.run_pipeline("unused.wav", fp.PipelineConfig(asr_backend="parakeet", device="cuda"))


def test_report_only_does_not_require_cuda(monkeypatch):
    monkeypatch.setattr(fp.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(fp, "_run_report_only", lambda *a: {"report": "ok"})
    assert fp.run_pipeline(
        "unused.wav", fp.PipelineConfig(asr_backend="parakeet", device="cuda", report_only=True)
    ) == {"report": "ok"}


def test_cuda_pipeline_forwards_device(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(fp.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(fp.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(fp, "convert_to_wav", lambda *a, **kw: tmp_path / "unused.wav")
    monkeypatch.setattr(
        fp, "apply_diarization", lambda *a, **kw: [{"start": 0, "end": 1, "speaker": 0}]
    )
    monkeypatch.setattr(pe, "load_parakeet", lambda *a, **kw: calls.append(kw))
    monkeypatch.setattr(pe, "transcribe_parakeet", lambda *a: [])
    monkeypatch.setattr(fp, "_run_postprocessing", lambda **kw: ({}, {}))
    fp.run_pipeline(
        "unused.wav", fp.PipelineConfig(out_dir=tmp_path, asr_backend="parakeet", device="cuda")
    )
    assert calls == [{"device": "cuda"}]


def test_cuda_refuses_native_cpu_fallback(capfd):
    import os

    def fallback(**kwargs):
        os.write(2, b"Failed to enable CUDA. Fallback to cpu\n")
        return object()

    with pytest.raises(RuntimeError, match="refusing CPU fallback"):
        pe._create_cuda_recognizer(fallback)
    assert "Failed to enable CUDA" in capfd.readouterr().err


def test_cuda_preserves_native_exception_and_stderr(capfd):
    import os

    def broken(**kwargs):
        os.write(2, b"native initialization error\n")
        raise ValueError("initialization failed")

    with pytest.raises(ValueError, match="initialization failed"):
        pe._create_cuda_recognizer(broken)
    os.write(2, b"stderr restored\n")
    assert capfd.readouterr().err == "native initialization error\nstderr restored\n"


@pytest.mark.gpu
@pytest.mark.integration
def test_real_cuda_execution_profile(monkeypatch, tmp_path):
    import gc
    import json
    import os

    root = os.environ.get("ASR_TEST_PARAKEET_MODEL")
    if not root or os.environ.get("ASR_TEST_PARAKEET_DEVICE") != "cuda":
        pytest.skip("Set ASR_TEST_PARAKEET_MODEL and ASR_TEST_PARAKEET_DEVICE=cuda")
    import sherpa_onnx

    factory = sherpa_onnx.OfflineRecognizer.from_transducer
    config = tmp_path / "cuda.conf"
    config.write_text(f"ProfilingFilePrefix={tmp_path / 'cuda_trace'}\n")

    def profiled(**kwargs):
        assert kwargs["provider"] == "cuda"
        kwargs["provider"] = f"cuda:{config}"
        return factory(**kwargs)

    monkeypatch.setattr(sherpa_onnx.OfflineRecognizer, "from_transducer", profiled)
    with sf.SoundFile(Path(__file__).parent / "data" / "test.wav") as audio:
        sf.write(tmp_path / "sample.wav", audio.read(20 * 16000), 16000)
    # Exercise the actual orchestrator, ASR and exports. Isolate external
    # diarization/report services so this checks CUDA without gated models/API calls.
    monkeypatch.setattr(fp, "apply_diarization", lambda *a, **kw: [
        {"start": 0, "end": 20, "speaker": 0}])
    monkeypatch.setattr(fp, "_run_postprocessing", lambda **kw: ({}, {}))
    result = fp.run_pipeline(tmp_path / "sample.wav", fp.PipelineConfig(
        asr_backend="parakeet", device="cuda", parakeet_model_dir=Path(root),
        out_dir=tmp_path / "outputs"))
    assert result["asr"] and result["asr"][0]["text"]
    assert json.loads(Path(result["json"]).read_text())["segments"][0]["speaker"] == 0
    gc.collect()
    providers = [
        event.get("args", {}).get("provider")
        for trace in tmp_path.glob("cuda_trace*.json")
        for event in json.loads(trace.read_text())
    ]
    assert "CUDAExecutionProvider" in providers
