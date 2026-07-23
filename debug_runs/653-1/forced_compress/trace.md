# Forced compression

1. The harness manually decoded 300 source frames from 0.0s to 300.0s. The model did not choose or call a tool.

2. It passed the recorded input tensor through the existing vision tower and unchanged FlashVID compressor.

3. FlashVID reduced 6750 visual tokens to 5895 against a budget of 5760.

4. Boundary tensor shapes, finite checks, kept indices, and M-RoPE shapes are in `tensor_shapes.json`.

5. Sampled source frames are in `visualizations/sampled_source_frames.png`.

6. `prompt.txt` and `inserted_visual_text_context` show the exact text around the inserted visual representation.

7. The harness generated once, saved the raw output, parsed it, and stopped.
