import json
import os
from typing import Dict, Generator, Iterable


def iter_json_lines(filepath: str) -> Generator[Dict, None, None]:
    with open(filepath) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def dump_json_lines(rows: Iterable[Dict], filepath: str, mode: str = "w", **json_kwargs) -> None:
    with open(filepath, mode) as f:
        for row in rows:
            f.write(json.dumps(row, **json_kwargs) + "\n")


def load_json(filepath: str) -> Dict:
    with open(filepath) as f:
        return json.load(f)


def dump_json(data: Dict, filepath: str, mode: str = "w", **json_kwargs) -> None:
    with open(filepath, mode) as f:
        json.dump(data, f, **json_kwargs)


def dump_yaml(data: Dict, filepath: str, indent: int = 4, offset: int = 2) -> None:
    from ruamel.yaml import YAML

    yaml_obj = YAML()
    yaml_obj.indent(sequence=indent, offset=offset)
    with open(filepath, "w") as f:
        yaml_obj.dump(data, f)


def load_yaml(filepath: str) -> Dict:
    import yaml

    with open(filepath) as f:
        return yaml.safe_load(f)


def dump_text(text: str, filepath: str, mode: str = "w", exist_ok: bool = True) -> None:
    if not exist_ok and os.path.exists(filepath):
        raise FileExistsError(f"expected path to not exist: {filepath}")
    with open(filepath, mode) as f:
        f.write(text)


def rmfile(filepath: str) -> None:
    os.remove(filepath)


def check_path_exists(path: str) -> bool:
    return os.path.exists(path)


def makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)
