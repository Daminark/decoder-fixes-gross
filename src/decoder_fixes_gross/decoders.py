"""Decoders behind one interface: a syndrome in, a correction over all model columns out.

The interface is a draft: it is fixed once a second decoder family exists.
The first decoder is Relay-BP-S of Müller et al. [1]_.

References
----------
.. [1] T. Müller, T. Alexander, M. E. Beverland, M. Bühler, B. R. Johnson, T. Maurer and
   D. Vandeth, "Improved belief propagation is sufficient for real-time decoding of quantum
   memory", arXiv:2506.01779 (2025).
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import relay_bp
from numpy.typing import NDArray

from decoder_fixes_gross.dem import Dem


@dataclass(frozen=True, eq=False)
class Decoding:
    """Result of one decode.

    Attributes
    ----------
    correction : numpy.ndarray
        uint8 0/1 vector over all columns of the model the decoder was built from.
    iterations : int or None
        Iterations the decoder ran, or None for a decoder that has no iteration count.
    """

    correction: NDArray[np.uint8]
    iterations: int | None


class Decoder(Protocol):
    """A decoder built from one model.

    ``decode`` takes a uint8 0/1 syndrome of length n_detectors and returns a `Decoding`
    whose correction has length n_columns of that model. A decoder may hold state between
    calls, so the result of a call can depend on the calls before it.
    """

    def decode(self, syndrome: NDArray[np.uint8]) -> Decoding:
        """Return the correction for ``syndrome``."""
        ...


@dataclass(frozen=True)
class RelaySettings:
    """Settings of Relay-BP-S, named as the arguments of ``relay_bp.RelayDecoderF64``.

    Attributes
    ----------
    gamma0 : float
        Uniform memory strength of the first leg.
    pre_iter : int
        Iterations of the first leg.
    num_sets : int
        Legs after the first; each draws its memory strengths uniformly from
        ``gamma_dist_interval``.
    set_max_iter : int
        Iterations of each later leg.
    gamma_dist_interval : tuple of float
        Interval the later legs' memory strengths are drawn from.
    stop_nconv : int
        Number of syndrome-valid estimates after which decoding stops.
    """

    gamma0: float
    pre_iter: int
    num_sets: int
    set_max_iter: int
    gamma_dist_interval: tuple[float, float]
    stop_nconv: int


# Relay-BP-1 for the gross code, arXiv:2506.01779, the settings paragraph and Fig. 2.
RELAY_BP_BASELINE = RelaySettings(
    gamma0=0.125,  # "uniform memory strength of 0.125 for the gross and two-gross codes"
    pre_iter=80,  # first leg, T_0 = 80
    num_sets=300,  # R = 301 legs in all
    set_max_iter=60,  # later legs, T_r = 60
    gamma_dist_interval=(-0.24, 0.66),  # Fig. 2, the interval circled for the gross code
    stop_nconv=1,  # Relay-BP-1: stop at the first syndrome-valid estimate
)


class RelayBP:
    """Relay-BP-S of Müller et al. (arXiv:2506.01779), from the ``relay-bp`` package.

    Facts about ``relay_bp.RelayDecoderF64`` 0.2.0 on the gross-code model at p = 1e-3,
    measured when this wrapper was written:

    - ``decode_detailed`` gives an iteration count summed over legs, in which a leg that
      converges on its k-th iteration counts k - 1: in ``decode_inner`` of
      ``crates/relay_bp/src/bp/relay.rs`` (relay_bp-0.2.0 sdist), the counter increment
      follows the convergence check, which exits the loop first. With
      ``stop_nconv`` = 1 at most one leg converges, so the iterations run are the count
      plus one when the decode succeeds; `decode` reports that. A decode that finds no
      syndrome-valid estimate runs all 80 + 300 x 60 = 18,080 iterations.
    - Repeated decodes on one object are not independent: the object's random stream
      advances, so the same syndrome can decode differently on each call. A new object
      with the same seed reproduces its first decode exactly.
    - An object cannot be reseeded; a new seed means a new object.
    - Construction takes about 2.3 ms.
    - The read-only arrays of a `Dem` are accepted without copying.

    Parameters
    ----------
    dem : Dem
        Model whose ``h`` and ``priors`` the decoder uses.
    seed : int
        Seed of the decoder's random stream, in [0, 2**64).
    settings : RelaySettings, optional
        Defaults to `RELAY_BP_BASELINE`.

    Raises
    ------
    TypeError
        If ``seed`` is not an integer.
    ValueError
        If ``seed`` is not in [0, 2**64).
    """

    def __init__(self, dem: Dem, *, seed: int, settings: RelaySettings = RELAY_BP_BASELINE) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int | np.integer):
            raise TypeError(f"seed must be an integer; got {seed!r}")
        if not 0 <= seed < 2**64:
            raise ValueError(f"seed must be in [0, 2**64); got {seed}")
        self._n_detectors = dem.h.shape[0]
        self._decoder = relay_bp.RelayDecoderF64(
            dem.h,
            error_priors=dem.priors,
            seed=int(seed),
            gamma0=settings.gamma0,
            pre_iter=settings.pre_iter,
            num_sets=settings.num_sets,
            set_max_iter=settings.set_max_iter,
            gamma_dist_interval=settings.gamma_dist_interval,
            stop_nconv=settings.stop_nconv,
        )

    def decode(self, syndrome: NDArray[np.uint8]) -> Decoding:
        """Return the correction for ``syndrome`` and the number of iterations run.

        Raises
        ------
        ValueError
            If ``syndrome`` is not a 0/1 vector of length n_detectors.
        """
        detectors = np.ascontiguousarray(syndrome, dtype=np.uint8)
        if detectors.shape != (self._n_detectors,) or (detectors > 1).any():
            raise ValueError(
                f"syndrome must be a 0/1 vector of length {self._n_detectors}; "
                f"got shape {np.shape(syndrome)}"
            )
        result = self._decoder.decode_detailed(detectors)
        # The converging iteration is not counted by the package; exact for stop_nconv = 1.
        iterations = int(result.iterations) + int(result.success)
        return Decoding(
            correction=np.asarray(result.decoding, dtype=np.uint8), iterations=iterations
        )
