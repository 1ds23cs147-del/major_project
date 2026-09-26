# Demo Image Pairs

Each capture ID identifies one RGB/thermal instance. Upload the files from one
row together and send the row's `capture_id` with the request. Do not combine
the RGB image from one row with the thermal image from another row.

The files are kept in `webapp/uploads/` because they are already part of the
application's captured-image store.

| capture_id | RGB file | Thermal file |
| --- | --- | --- |
| `cap_1788093931850_3jinv5` | `cap_1788093931850_3jinv5_rgb.jpg` | `cap_1788093931850_3jinv5_thermal.jpg` |
| `test_cap_1` | `test_cap_1_rgb.jpg` | `test_cap_1_thermal.jpg` |

The thermal model remains a classifier: it returns a fire/no-fire score and
does not produce object-detection boxes. RGB detection and thermal
classification are separate operations in the multimodal demo.

The webapp exposes these pairs through `/api/demo-pairs`. Open the webapp at
`http://127.0.0.1:5001`, choose **Use this pair**, and then click **Analyze
Scene**. The selected capture ID is preserved in the request.