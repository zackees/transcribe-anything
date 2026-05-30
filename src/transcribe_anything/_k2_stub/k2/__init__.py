"""Stub `k2` package shipped with transcribe-anything.

Why this exists
---------------
``speechbrain>=1.0`` ships an integration sub-package at
``speechbrain.integrations.k2_fsa``. Its ``__init__.py`` eagerly does::

    import k2  # noqa
    ...
    lazy_export_all(__file__, __name__, export_subpackages=True)

If the real ``k2`` (https://github.com/k2-fsa/k2) is not installed, that
import raises ``ImportError`` and the speechbrain lazy-loader propagates
it — which crashes pyannote diarization in the insane backend on any
platform where ``k2`` has no installable wheel (notably Windows, where
``k2`` has no Windows wheels at all). See issue #69.

What it does
------------
Provides an empty ``k2`` package that satisfies the ``import k2`` line.
``lazy_export_all`` in speechbrain's k2_fsa init only registers
sub-modules for *lazy* loading; nothing in the integration is actually
exercised at import time, so an empty stub is sufficient to keep the
loader from crashing.

When the *real* ``k2`` is installed in the venv (e.g. Linux + CUDA
configurations where k2 wheels are available), Python's normal import
resolution finds the real package first — this stub does not shadow it
unless the stub directory is explicitly the first entry on the path.
"""
