"""Container for the ``k2`` stub package.

This directory exists only so that ``_k2_stub`` gets picked up as a
``transcribe_anything`` sub-package by ``setuptools.packages.find`` and
the ``k2/`` stub it contains is included in built wheels.

At runtime we put the *parent* of ``k2/`` (i.e. this directory's path)
on ``PYTHONPATH`` so the venv's Python finds ``k2`` as a top-level
package. See ``_k2_stub_root`` and ``_prepare_subprocess_env`` in
``transcribe_anything.insanely_fast_whisper`` and issue #69.
"""
