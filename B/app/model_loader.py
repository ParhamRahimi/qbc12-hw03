"""app.model_loader — wraps the HW3_A bundle's predict.py."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_DEFAULT_BUNDLE_IN_IMAGE = "/app/bundle"
_DEV_BUNDLE = str(
    Path(__file__).resolve().parent.parent.parent / "HW3_A" / "bundle"
)


def _resolve_bundle_dir() -> Path:
    env = os.getenv("BUNDLE_DIR", "").strip()
    if env:
        p = Path(env).resolve()
        if p.exists():
            return p
    if Path(_DEFAULT_BUNDLE_IN_IMAGE).exists():
        return Path(_DEFAULT_BUNDLE_IN_IMAGE)
    if Path(_DEV_BUNDLE).exists():
        return Path(_DEV_BUNDLE)
    raise FileNotFoundError(
        f"Bundle not found. Set BUNDLE_DIR env var. "
        f"Tried: {_DEFAULT_BUNDLE_IN_IMAGE}, {_DEV_BUNDLE}"
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_manifest(bundle_dir: Path) -> tuple[bool, str]:
    manifest_path = bundle_dir / "MANIFEST.json"
    if not manifest_path.exists():
        return False, "MANIFEST.json not found"
    manifest = json.loads(manifest_path.read_text())
    files = manifest.get("files", {})
    ok = 0
    for rel, expected in files.items():
        if expected.startswith("REPLACE"):
            return False, f"Placeholder SHA in MANIFEST: {rel}"
        fpath = bundle_dir / rel
        if not fpath.exists():
            return False, f"File in manifest but missing on disk: {rel}"
        actual = _sha256(fpath)
        if actual != expected:
            return False, f"SHA mismatch for {rel}: expected {expected[:16]}..., got {actual[:16]}..."
        ok += 1
    return True, f"{ok} files OK"


@dataclass
class LoadState:
    loaded: bool = False
    error: Optional[str] = None
    bundle_dir: Optional[Path] = None
    manifest_ok: Optional[bool] = None
    manifest_msg: Optional[str] = None


@dataclass
class ModelService:
    state: LoadState = field(default_factory=LoadState)
    predictor: Optional[object] = None
    metadata: dict = field(default_factory=dict)

    def load(self) -> None:
        try:
            bundle_dir = _resolve_bundle_dir()
            self.state.bundle_dir = bundle_dir
            ok, msg = _verify_manifest(bundle_dir)
            self.state.manifest_ok = ok
            self.state.manifest_msg = msg
            if not ok:
                self.state.error = f"MANIFEST verification failed: {msg}"
                return

            model_dir = bundle_dir / "model"
            if not model_dir.exists():
                self.state.error = f"model/ directory not found in bundle: {model_dir}"
                return

            if str(bundle_dir) not in sys.path:
                sys.path.insert(0, str(bundle_dir))
            from predict import BundlePredictor  # type: ignore  # noqa: E402
            self.predictor = BundlePredictor(bundle_dir=model_dir)

            meta_path = bundle_dir / "metadata.json"
            if meta_path.exists():
                self.metadata = json.loads(meta_path.read_text())

            self.state.loaded = True
        except Exception as e:
            self.state.error = str(e)

    def require_predictor(self) -> object:
        if not self.state.loaded or self.predictor is None:
            raise RuntimeError("Model not loaded")
        return self.predictor

    def info(self) -> dict:
        return {
            "bundle_loaded": self.state.loaded,
            "bundle_dir": str(self.state.bundle_dir) if self.state.bundle_dir else "",
            "manifest_ok": self.state.manifest_ok,
            "message": self.state.manifest_msg or self.state.error or "",
        }