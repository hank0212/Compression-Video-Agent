# Baseline single turn

1. The harness decoded the initial whole-video view and saved `visualizations/initial.png`.

2. It combined the unchanged question, initial visual input, and answer instruction.

3. It performed exactly one model generation and parsed that output directly.

4. It stopped. No tool loop and no finalizer are reachable in this mode.
