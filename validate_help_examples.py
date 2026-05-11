#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Validate the beginner-facing GR_python help examples.

Usage:
    python validate_help_examples.py
    python gr_help.py validate
"""

from gr_help import validate_examples


if __name__ == "__main__":
    raise SystemExit(0 if validate_examples() else 1)
