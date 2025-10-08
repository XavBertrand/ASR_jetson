from dataclasses import dataclass
import yaml

@dataclass
class AppCfg:
    device: str = "cuda"
    batch_size: int = 1
    vad: str = "silero"
    asr: str = "fasterwhisper"

def load_config(path: str|None) -> AppCfg:
    if not path: return AppCfg()
    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    return AppCfg(**d.get("runtime", {}))
