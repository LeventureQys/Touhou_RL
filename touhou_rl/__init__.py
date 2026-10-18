"""Touhou RL framework (TH08 / Imperishable Night).

The package is split into three layers:

*  ``winapi`` / ``memory`` / ``input`` / ``capture`` -- OS boundary (Win32, read-only memory).
*  ``state`` / ``session`` / ``menu`` / ``obs`` / ``reward`` / ``env`` -- game boundary.
*  ``models`` / ``dataset`` / ``bc`` / ``ppo`` / ``evaluate`` / ``play`` -- training boundary.

The observation and action contracts are documented in
``Document/Update/Alpha/v1.0.2/框架接口契约.md``; this package implements them.
"""

__version__ = "1.0.2"
