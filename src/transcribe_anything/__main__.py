#!/usr/bin/env python3
"""
Main entry point for transcribe_anything package when run as a module.
This allows the package to be executed with: python -m transcribe_anything
"""

from transcribe_anything._cmd import main

if __name__ == "__main__":
    exit(main())
