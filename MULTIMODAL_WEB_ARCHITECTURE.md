# Multimodal Forest-Fire Web Application Architecture

## Goal

Accept three aligned versions of the same scene—RGB, thermal/infrared, and NIR—and produce a high-confidence forest-fire alert. The system checks the inputs in this order:

```text
RGB image -> thermal image -> NIR image -> fused decision -> optional SOS alert
```

The model must not treat three unrelated uploads as one scene. Each upload set needs a shared `capture_id`, nearly identical capture time, and the same camera viewpoint. For best results, use calibrated sensors mounted together; otherwise image registration is required before inference.

## User flow

1. User creates an incident and uploads the RGB image.
2. The server runs the RGB fire/smoke detector immediately and displays its boxes and confidence.
3. User uploads the thermal version of that same image. The server aligns it to RGB and runs the thermal detector.
4. User uploads the NIR version. The server aligns it to RGB and runs the NIR detector.
5. The decision service fuses all valid modality scores, applies false-alert rules, and returns `No fire`, `Needs review`, or `Confirmed fire`.
6. Only `Confirmed fire` can create an SOS alert, subject to confirmation and rate limits.

## Architecture

```text
Browser / mobile web app
  |
  |-- upload RGB, thermal, NIR images + capture metadata
  v
Web API
  |-- authentication and request validation
  |-- object storage (original files, encrypted at rest)
  |-- incident database (capture_id, status, scores, audit log)
  v
Background job queue
  |
  |-- RGB inference worker       -> RGB fire/smoke score and boxes
  |-- thermal inference worker   -> thermal fire score and boxes
  |-- NIR inference worker       -> NIR fire/context score and boxes
  |-- alignment/quality worker   -> checks that all images match one scene
  v
Decision / fusion service
  |-- sequential verification rules
  |-- calibrated confidence fusion
  |-- temporal and duplicate-alert checks
  v
Web dashboard + alert service
  |-- annotated evidence and confidence explanation
  |-- human approval where required
  `-- SMS/voice SOS provider -> configured emergency contacts
```

## Model design

Use three independently trained models initially. This is safer than trying to force a single RGB model to understand all sensor types.

| Stage | Input | Model output | Required training data |
|---|---|---|---|
| 1 | RGB | fire/smoke boxes and confidence | RGB fire/smoke labels |
| 2 | Thermal/IR | fire boxes and confidence | Thermal/IR fire labels |
| 3 | NIR | fire/context boxes and confidence | NIR images with fire/no-fire labels |

The current project has RGB and some labeled infrared/thermal fire data. The current `nirscene0` images have no fire/no-fire labels, so they must **not** be used to claim NIR fire detection until they are labelled. They can later support a separate forest-context model if their ground truth is available.

### Sequential decision logic

```text
1. RGB score < RGB_REVIEW_THRESHOLD
      -> No fire (unless an operator explicitly requests all modalities)

2. RGB score >= RGB_REVIEW_THRESHOLD
      -> run / require thermal verification

3. Thermal score >= THERMAL_CONFIRM_THRESHOLD
      -> run / require NIR verification

4. RGB + thermal + NIR are aligned and pass quality checks
      -> calculate fused score

5. Fused score >= SOS_THRESHOLD and boxes overlap in the same scene
      -> Confirmed fire; allow SOS alert
   Otherwise
      -> Needs review or No fire
```

Suggested initial thresholds (tune only on validation data):

```text
RGB_REVIEW_THRESHOLD       = 0.40
THERMAL_CONFIRM_THRESHOLD  = 0.50
NIR_CONFIRM_THRESHOLD      = 0.50
SOS_THRESHOLD              = 0.75
```

### Confidence fusion

Do not simply average raw model confidences: RGB, thermal, and NIR models can be calibrated differently. First calibrate each model on a validation set (for example, temperature scaling), then fuse:

```text
fused_score = 0.40 * calibrated_rgb
            + 0.40 * calibrated_thermal
            + 0.20 * calibrated_nir
```

Increase or decrease these weights only after evaluating a held-out multimodal validation set. If NIR quality is poor or the NIR model is not trained, mark NIR as unavailable and do not fabricate a NIR score.

## False-alert controls

- Reject a modality if it is blurry, overexposed, missing metadata, or cannot be aligned to RGB.
- Require RGB and thermal detections to overlap after image registration.
- Require at least two nearby detections over time for live video (for example, 10-second samples).
- Deduplicate alerts by camera, location, and a short time window.
- Provide `Needs review` for ambiguous cases rather than automatically sending SOS.
- Record the three annotated images and model scores with every decision for auditing.

## SOS-alert design

The web application should never send unlimited automatic messages. Use this flow:

```text
Confirmed fire -> alert-rate-limit check -> optional operator confirmation
-> send SMS/voice alert -> save provider message ID and delivery status
```

The alert should contain the camera/location, timestamp, fused confidence, and a secure link to the annotated RGB/thermal/NIR evidence. Store mobile numbers encrypted, require authenticated configuration, and obtain explicit consent from every contact. Use an approved SMS/voice provider and official local emergency procedures; do not assume an SMS is a substitute for emergency services.

## Recommended implementation phases

1. Build the RGB upload, RGB detector, dashboard, and result storage.
2. Add thermal upload, registration, and thermal confirmation model.
3. Label or obtain labelled NIR fire/no-fire data, then train the NIR model.
4. Add calibrated fusion and validation-set threshold tuning.
5. Add an operator-reviewed SMS alert workflow.
6. Add live camera ingestion and temporal confirmation.

## Evaluation

Evaluate each modality separately and the final fused decision on an untouched test set. Report:

- Per-model precision, recall, F1, mAP@50, and mAP@50-95.
- Final binary alert precision, recall, F1, and false-alert rate.
- Metrics for RGB-only, RGB+thermal, and RGB+thermal+NIR decisions.
- Latency from upload to decision and SMS delivery status.

The final system should be optimized for high recall, while the thermal/NIR verification, alignment checks, and confirmation rules control false alerts.
