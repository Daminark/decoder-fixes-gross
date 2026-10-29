"""Files of a decoding run: the configuration record and the shards of per-orbit outcomes.

A run directory holds ``config.json`` and one ``shard_<k>.npz`` per shard, k zero-padded to
four digits. Every file is written under a temporary name and renamed when complete, so the
directory never holds a partial file under a final name.
"""

import dataclasses
import json
import os
import platform
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import IO, Any, Self

import numpy as np
from numpy.typing import NDArray

from decoder_fixes_gross.inject import Outcome
from decoder_fixes_gross.symmetry import N_TRANSLATIONS

CONFIG_NAME = "config.json"
# Distributions whose versions can change an outcome or a timing.
_PACKAGES = ("numpy", "scipy", "stim", "relay-bp", "ldpc", "decoder-fixes-gross")
_SHARD_DTYPES = {
    "translates": np.uint8,
    "outcomes": np.uint8,
    "micros": np.uint32,
    "iterations": np.int32,
}


def shard_name(index: int) -> str:
    """Return the file name of shard ``index``."""
    return f"shard_{index:04d}.npz"


def commit_hash(path: Path) -> str:
    """Return the HEAD commit of the git checkout containing ``path``.

    Returns
    -------
    str
        The full hash, followed by ``-dirty`` when tracked files differ from HEAD; "unknown"
        when git is not installed or ``path`` is not inside a checkout.
    """
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True, check=False
        )
        if head.returncode != 0:
            return "unknown"
        diff = subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=path, check=False)
    except FileNotFoundError:
        return "unknown"
    return head.stdout.strip() + ("-dirty" if diff.returncode != 0 else "")


def package_versions() -> dict[str, str]:
    """Return the Python version and the installed versions of the packages a run depends on."""
    return {"python": platform.python_version()} | {name: version(name) for name in _PACKAGES}


@dataclass(frozen=True, eq=False)
class Shard:
    """Per-orbit results of one shard, in the order of the orbits in their list.

    Attributes
    ----------
    decoder_seed : int
        Seed of the shard's decoder, in [0, 2**64).
    translates : numpy.ndarray
        uint8 of shape [n]; the group element in [0, 72) applied to each representative.
    outcomes : numpy.ndarray
        uint8 of shape [n]; the `Outcome` code of each decode.
    micros : numpy.ndarray
        uint32 of shape [n]; the decode time in microseconds.
    iterations : numpy.ndarray
        int32 of shape [n]; the decoder's iteration count, or -1 if it has none.

    The arrays are made read-only in place.

    Raises
    ------
    ValueError
        If a dtype, shape or range condition above does not hold.
    """

    decoder_seed: int
    translates: NDArray[np.uint8]
    outcomes: NDArray[np.uint8]
    micros: NDArray[np.uint32]
    iterations: NDArray[np.int32]

    def __post_init__(self) -> None:
        """Check dtypes, shapes and ranges, then freeze the arrays."""
        if not 0 <= self.decoder_seed < 2**64:
            raise ValueError(f"decoder_seed must be in [0, 2**64), got {self.decoder_seed}")
        arrays = self._arrays()
        n = self.translates.shape
        for name, array in arrays.items():
            expected = _SHARD_DTYPES[name]
            if array.dtype != expected or array.ndim != 1 or array.shape != n:
                raise ValueError(
                    f"{name} must be {np.dtype(expected)} of shape {n}, "
                    f"got {array.dtype} {array.shape}"
                )
        if np.any(self.translates >= N_TRANSLATIONS):
            raise ValueError(f"translates must be in [0, {N_TRANSLATIONS})")
        if np.any(self.outcomes > max(Outcome)):
            raise ValueError(f"outcomes must be codes in [0, {max(Outcome)}]")
        if np.any(self.iterations < -1):
            raise ValueError("iterations must be counts or -1")
        for array in arrays.values():
            array.flags.writeable = False

    def __len__(self) -> int:
        """Return the number of orbits in the shard."""
        return len(self.translates)

    def save(self, path: Path) -> None:
        """Write the shard to ``path`` as an uncompressed ``.npz``, atomically."""

        def write(f: IO[bytes]) -> None:
            np.savez(
                f,
                decoder_seed=np.uint64(self.decoder_seed),
                translates=self.translates,
                outcomes=self.outcomes,
                micros=self.micros,
                iterations=self.iterations,
            )

        _write_atomic(path, write)

    @classmethod
    def load(cls, path: Path) -> Self:
        """Read a shard written by :meth:`save`.

        Raises
        ------
        ValueError
            If the file lacks a key or its arrays fail the checks of the class.
        """
        with np.load(path, allow_pickle=False) as data:
            missing = {"decoder_seed", *_SHARD_DTYPES} - set(data.files)
            if missing:
                raise ValueError(f"{path} lacks the keys {sorted(missing)}")
            return cls(
                decoder_seed=int(data["decoder_seed"]),
                translates=data["translates"],
                outcomes=data["outcomes"],
                micros=data["micros"],
                iterations=data["iterations"],
            )

    def _arrays(self) -> dict[str, NDArray[np.generic]]:
        return {
            "translates": self.translates,
            "outcomes": self.outcomes,
            "micros": self.micros,
            "iterations": self.iterations,
        }


@dataclass(frozen=True)
class ListRecord:
    """A fault list as a run records it.

    Attributes
    ----------
    name : str
        Name of the list in the run, such as ``u3``.
    content_hash : str
        :meth:`decoder_fixes_gross.faultlist.FaultList.content_hash` of the list.
    n_orbits : int
        Number of orbit representatives.
    weight : int
        Number of columns in each fault set.
    """

    name: str
    content_hash: str
    n_orbits: int
    weight: int


@dataclass(frozen=True)
class ShardSpec:
    """The orbits a shard covers: rows ``start`` to ``stop`` (exclusive) of one list.

    Attributes
    ----------
    index : int
        Position of the shard in the run, from 0.
    list_name : str
        Name of the list.
    start, stop : int
        Row range in the list.
    """

    index: int
    list_name: str
    start: int
    stop: int


@dataclass(frozen=True, eq=False)
class RunConfig:
    """Configuration record of a run, written to ``config.json`` in its directory.

    Attributes
    ----------
    decoder : str
        Name of the decoder.
    decoder_settings : Mapping
        The decoder's settings; JSON-serializable.
    p0 : float
        Physical error rate of the model the decoder is built from.
    run_seed : int
        Non-negative seed from which every shard's random stream is spawned.
    shard_size : int
        Maximum number of orbits in a shard; positive.
    lists : tuple of ListRecord
        The lists of the run, in the order they are sharded; names distinct.
    versions : Mapping
        Package versions, as from `package_versions`.
    commit : str
        Commit of the code, as from `commit_hash`.

    Raises
    ------
    ValueError
        If ``run_seed`` is negative, ``shard_size`` is not positive, or list names repeat.
    """

    decoder: str
    decoder_settings: Mapping[str, Any]
    p0: float
    run_seed: int
    shard_size: int
    lists: tuple[ListRecord, ...]
    versions: Mapping[str, str]
    commit: str

    def __post_init__(self) -> None:
        """Check the seed, the shard size and the list names."""
        if self.run_seed < 0:
            raise ValueError(f"run_seed must be non-negative, got {self.run_seed}")
        if self.shard_size < 1:
            raise ValueError(f"shard_size must be positive, got {self.shard_size}")
        names = [record.name for record in self.lists]
        if len(set(names)) != len(names):
            raise ValueError(f"list names must be distinct, got {names}")

    def shards(self) -> list[ShardSpec]:
        """Return the shards of the run: each list in order, cut into runs of ``shard_size``.

        A shard never covers two lists; only the last shard of a list can be short.
        """
        specs: list[ShardSpec] = []
        for record in self.lists:
            for start in range(0, record.n_orbits, self.shard_size):
                stop = min(start + self.shard_size, record.n_orbits)
                specs.append(ShardSpec(len(specs), record.name, start, stop))
        return specs

    def to_json(self) -> dict[str, Any]:
        """Return the record as it is stored: a dict of JSON values (tuples become lists)."""
        record = {
            "decoder": self.decoder,
            "decoder_settings": dict(self.decoder_settings),
            "p0": self.p0,
            "run_seed": self.run_seed,
            "shard_size": self.shard_size,
            "lists": [dataclasses.asdict(list_record) for list_record in self.lists],
            "versions": dict(self.versions),
            "commit": self.commit,
        }
        normalized: dict[str, Any] = json.loads(json.dumps(record))
        return normalized

    def differing_fields(self, other: "RunConfig") -> list[str]:
        """Return the names of the fields whose stored values differ from those of ``other``."""
        mine, theirs = self.to_json(), other.to_json()
        return [name for name in mine if mine[name] != theirs[name]]

    def save(self, run_dir: Path) -> None:
        """Write ``config.json`` in ``run_dir``, atomically."""
        text = json.dumps(self.to_json(), indent=2) + "\n"
        _write_atomic(run_dir / CONFIG_NAME, lambda f: f.write(text.encode()))

    @classmethod
    def load(cls, run_dir: Path) -> Self:
        """Read ``config.json`` from ``run_dir``.

        Raises
        ------
        ValueError
            If a field, at the top level or in a list record, is missing, or a field fails
            the checks of the class.
        """
        record = json.loads((run_dir / CONFIG_NAME).read_text(encoding="utf-8"))
        try:
            return cls(
                decoder=record["decoder"],
                decoder_settings=record["decoder_settings"],
                p0=record["p0"],
                run_seed=record["run_seed"],
                shard_size=record["shard_size"],
                lists=tuple(
                    ListRecord(
                        name=list_record["name"],
                        content_hash=list_record["content_hash"],
                        n_orbits=list_record["n_orbits"],
                        weight=list_record["weight"],
                    )
                    for list_record in record["lists"]
                ),
                versions=record["versions"],
                commit=record["commit"],
            )
        except KeyError as missing:
            raise ValueError(f"{run_dir / CONFIG_NAME} lacks the field {missing}") from None


def _write_atomic(path: Path, write: Callable[[IO[bytes]], object]) -> None:
    """Write a file through ``write`` under a temporary name, then rename it to ``path``."""
    partial = path.with_name(path.name + ".tmp")
    try:
        with partial.open("wb") as f:
            write(f)
            f.flush()
            # The bytes must reach the disk before the rename, or a crash could leave the
            # final name on a partial file. A rename lost to a crash only repeats the work.
            os.fsync(f.fileno())
        partial.replace(path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
