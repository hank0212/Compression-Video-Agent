# Forced crop

1. The harness manually decoded 128 frames from 0.0s to 300.0s. The model did not choose or call a tool.

2. Exact planned timestamps are in `frame_timestamps.json`; frame count and resolution are in `tensor_shapes.json`.

3. The crop montage is `visualizations/forced_crop.png`.

4. `prompt.txt` is the exact rendered model input after inserting the forced crop.

5. The harness generated once, saved the raw output, parsed it, and stopped.
