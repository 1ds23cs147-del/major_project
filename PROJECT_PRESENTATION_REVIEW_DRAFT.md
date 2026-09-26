# Forest Fire Detection and Multimodal Alert System

## Review Draft Based on `major project final - ppt format .pptx`

This is a review draft. No PDF has been created yet.

Replace the bracketed personal-information fields before final export:

- Student name: `[STUDENT NAME]`
- USN: `[USN]`
- Guide name and designation: `[GUIDE NAME, DESIGNATION]`
- Co-guide name and designation: `[CO-GUIDE NAME, DESIGNATION]`
- Department: `[DEPARTMENT NAME]`
- Institution: `[INSTITUTION NAME]`
- Presentation date: `25 September 2026`

---

## Slide 1 - Title Slide

### Forest Fire Detection and Multimodal Alert System

AI-based RGB fire/smoke detection with thermal fire/no-fire verification and a live FireWatch dashboard.

- Student: `[STUDENT NAME]`
- USN: `[USN]`
- Guide: `[GUIDE NAME, DESIGNATION]`
- Co-guide: `[CO-GUIDE NAME, DESIGNATION]`
- Department: `[DEPARTMENT NAME]`
- Institution: `[INSTITUTION NAME]`
- Date: `25 September 2026`

---

## Slide 2 - Abstract

Forest fires can spread rapidly before a human operator notices them. Manual camera monitoring is slow, difficult to scale, and vulnerable to false alarms caused by bright objects, sunlight, fog, and smoke-like patterns.

This project develops an AI-assisted forest-camera monitoring system. The RGB pipeline uses YOLO11n to detect the classes `fire` and `smoke` and return bounding boxes with confidence scores. A separate thermal model is used only as a `fire` / `no_fire` image classifier. The FireWatch web application displays live camera feeds, stores evidence, supports RGB/thermal pairing, and applies temporal and multimodal confirmation rules before an alert is raised.

The completed RGB baseline run reached approximately 68.7% precision, 57.6% recall, 63.7% mAP@50, and 38.6% mAP@50-95 on its validation split. Smoke detection was stronger than fire detection. The system is suitable as an operator-assistance prototype, but the current metrics are baseline evidence because the original split contains known duplicate and near-duplicate concerns. The thermal classifier also requires further calibration because it produced a high score for a cold-frame verification image.

---

# Slides 3-9 - Literature Survey

Use the following table format on each literature slide: author, year, methodology, findings, and limitations.

## Slide 3 - Literature Survey 1

| Author and year | Methodology | Findings | Limitations |
|---|---|---|---|
| Toreyin et al., 2006 | Video fire detection using temporal intensity, color, and motion characteristics | Demonstrated that fire-like motion and color changes can support early visual fire detection | Sensitive to illumination, camera motion, and fire-colored non-fire objects; dependent on hand-crafted rules |

**Connection to this project:** motivates automated visual monitoring but supports replacing fixed rules with learned RGB fire/smoke detection.

## Slide 4 - Literature Survey 2

| Author and year | Methodology | Findings | Limitations |
|---|---|---|---|
| Celik et al., 2007 | Statistical color-model analysis for fire-pixel identification in video | Color distributions can separate many fire pixels from ordinary scene content | Pixel-level color rules do not reliably localize objects or distinguish sunlight and red objects |

**Connection to this project:** motivates confidence-based object detection rather than color thresholding alone.

## Slide 5 - Literature Survey 3

| Author and year | Methodology | Findings | Limitations |
|---|---|---|---|
| Frizzi et al., 2016 | Convolutional neural network approach for video fire and smoke recognition | Deep features improve recognition of complex fire and smoke appearances | Performance depends on representative training data; small flames and domain changes remain difficult |

**Connection to this project:** supports transfer learning with a compact YOLO model for practical inference.

## Slide 6 - Literature Survey 4

| Author and year | Methodology | Findings | Limitations |
|---|---|---|---|
| Muhammad et al., 2018 | Deep convolutional neural network for image/video fire detection | Deep learning can improve automated fire recognition and reduce manual monitoring effort | Dataset bias and false alarms remain important; detection quality is affected by scene and camera conditions |

**Connection to this project:** motivates measuring precision and recall separately instead of reporting accuracy alone.

## Slide 7 - Literature Survey 5

| Author and year | Methodology | Findings | Limitations |
|---|---|---|---|
| Sharma et al., 2017 | Image-based forest-fire analysis using visual features and machine learning | Forest imagery can provide useful early warning signals | Outdoor conditions, haze, vegetation color, and camera viewpoint create domain-shift problems |

**Connection to this project:** motivates deployment on forest-facing cameras and explicit acknowledgement that the model is not a forest-scene classifier.

## Slide 8 - Literature Survey 6

| Author and year | Methodology | Findings | Limitations |
|---|---|---|---|
| Infrared/thermal fire-detection studies, 2018-2022 | Thermal hotspot analysis and infrared image classification | Thermal information can provide physical evidence when visible flame is hidden or ambiguous | Thermal sensors require calibration; hot non-fire objects can create false positives; bounding-box labels are not always available |

**Connection to this project:** motivates keeping thermal as a separate fire/no-fire classifier and using it as supporting evidence rather than forcing it into RGB object detection.

## Slide 9 - Literature Survey 7

| Author and year | Methodology | Findings | Limitations |
|---|---|---|---|
| Recent multimodal fire-monitoring systems, 2020-2024 | Late fusion of RGB, thermal, and sometimes NIR scores with temporal alert rules | Combining complementary sensors can reduce uncertainty and support human review | Fusion is only trustworthy when modalities belong to the same scene and are calibrated; unrelated images can create misleading confidence |

**Research gap identified:** a practical prototype must combine detection, thermal verification, capture-instance tracking, temporal confirmation, evidence storage, and human review in one workflow.

> Citation details for the survey should be checked against the final bibliography before academic submission. The references below provide the starting list.

---

## Slide 10 - Problem Statement and Objectives

### Problem statement

Forest-camera operators must monitor large volumes of video manually. Small or distant fire/smoke regions may be missed, while bright objects, sunlight, and thermal hotspots can produce false alarms. A practical system must detect visible fire/smoke, use thermal evidence separately, preserve the evidence used for each decision, and avoid treating unrelated images as one event.

### Objectives

1. Detect visible `fire` and `smoke` regions in RGB camera frames using YOLO11n.
2. Classify thermal images as `fire` or `no_fire` without using the thermal model for object-detection boxes.
3. Pair RGB and thermal images using a shared capture ID and camera pairing metadata.
4. Provide a live FireWatch dashboard with camera feeds, annotated evidence, confidence, and review status.
5. Reduce false alerts through confidence thresholds, temporal confirmation, thermal verification, and human review.

---

# Slides 11-12 - Existing System

## Slide 11 - Existing System Architecture

```text
Forest camera or phone stream
          |
          v
Human watches live video
          |
          v
Manual identification of fire/smoke
          |
          v
Phone call or manual emergency response
```

Typical existing approaches rely on:

- Continuous human monitoring
- Basic motion or color thresholds
- Separate camera feeds without synchronized evidence
- Manual confirmation after an operator notices a possible fire
- No persistent record of model confidence or annotated evidence

## Slide 12 - Existing System Drawbacks

- High dependence on operator attention and response time
- Difficult to monitor many cameras continuously
- Color-based methods confuse fire-colored objects with fire
- Smoke can be subtle, distant, or partially occluded
- Thermal and RGB evidence may not be aligned
- No consistent confidence threshold or temporal confirmation
- Alerts may be difficult to audit after an incident
- A detector trained on one environment may not generalize to another

---

# Slides 13-16 - Proposed System

## Slide 13 - Proposed System Architecture

```text
RGB camera ---------------------> RGB YOLO detector
                                      |
                                      | fire/smoke boxes + confidence
                                      v
Thermal camera -> thermal classifier (fire/no_fire score)
                                      |
                                      v
                         Capture ID and pair validation
                                      |
                                      v
                         Temporal/fusion decision service
                                      |
                 +--------------------+--------------------+
                 |                                         |
             No fire / review                         Confirmed fire
                 |                                         |
                 v                                         v
           Dashboard record                    Evidence + operator alert
```

The project also contains a multimodal web architecture for optional NIR support. NIR is not treated as a trained fire detector until reliable labels are available.

## Slide 14 - Proposed System Modules

1. **Dataset preparation:** prepares RGB detection and thermal classification data.
2. **RGB detector:** YOLO11n detects `fire` and `smoke` with bounding boxes.
3. **Thermal classifier:** YOLO11n classification model returns thermal `fire` / `no_fire` confidence only.
4. **Capture pairing:** RGB and thermal uploads share a `capture_id`; live cameras are paired through camera metadata.
5. **Fusion and temporal rules:** combines evidence and requires repeated convincing frames before a live alert.
6. **FireWatch dashboard:** provides live wall, camera configuration, classifier workflow, alerts, and history.
7. **Evidence store:** records scores, capture IDs, timestamps, and annotated snapshots for review.

## Slide 15 - Algorithms Used

### RGB object detection

- YOLO11n transfer learning
- Classes: `fire`, `smoke`
- Bounding-box confidence and class scores
- Image size used for the completed run: 320
- Optimizer: AdamW
- Augmentation: mosaic, horizontal flip, HSV variation, scale, and translation

### Thermal classification

- YOLO11n classification model
- Classes: `fire`, `no_fire`
- Input image classification only
- No thermal bounding-box output

### Decision logic

- RGB-only detection is treated as visible evidence.
- Thermal-only output is treated as a hotspot/fire classification and needs context.
- RGB and thermal evidence should represent the same capture instance.
- Repeated live detections reduce one-frame false alarms.
- Ambiguous cases are routed to review instead of automatic dispatch.

## Slide 16 - Same-Instance Pairing and Demo Design

A valid multimodal capture must contain:

- One shared `capture_id`
- Nearly matching capture time
- The same physical scene and camera viewpoint
- RGB and thermal files selected from the same capture

The project now has two verified demo pairs documented in `webapp/demo_pairs/README.md`:

| Capture ID | RGB | Thermal |
|---|---|---|
| `cap_1788093931850_3jinv5` | `cap_1788093931850_3jinv5_rgb.jpg` | `cap_1788093931850_3jinv5_thermal.jpg` |
| `test_cap_1` | `test_cap_1_rgb.jpg` | `test_cap_1_thermal.jpg` |

A shared ID provides traceability; it cannot prove that unrelated files show the same physical moment. The demo must therefore use known matched pairs.

---

# Slides 17-20 - Results and Discussion

## Slide 17 - Completed RGB Training Results

Completed run:

- Run: `project_data/runs/forest_fire/fasdd_real_smoke_30ep`
- Training: 30/30 epochs completed
- Final best checkpoint: `models/rgb_fire_smoke_best.pt`
- RGB classes: `fire`, `smoke`

| Metric | Final validation result |
|---|---:|
| Precision | 68.7% |
| Recall | 57.6% |
| mAP@50 | 63.7% |
| mAP@50-95 | 38.6% |

## Slide 18 - Per-Class Results

| Class | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---:|---:|---:|---:|
| Fire | 63.3% | 53.0% | 58.0% | 32.0% |
| Smoke | 74.1% | 62.2% | 69.3% | 45.1% |
| Overall | 68.7% | 57.6% | 63.7% | 38.6% |

### Discussion

- Smoke detection is stronger than fire detection.
- Small or distant flames are harder to detect than larger smoke regions.
- A matchstick flame is an especially difficult case because it occupies very few pixels.
- These results are baseline validation results, not final leakage-free evidence.

## Slide 19 - Thermal Classifier Results and Limitation

Thermal model:

- Checkpoint: `models/thermal_fire_no_fire_best.pt`
- Classes: `fire`, `no_fire`
- Training reached epoch 18
- Thermal model is classification-only and does not output bounding boxes

Observed issue:

- The original validation result was effectively perfect.
- FireWatch verification gave a cold frame approximately 0.90 fire confidence.
- Therefore, the thermal result is not accepted as production evidence yet.
- Calibration, hard negative testing, and additional clean validation are required.

## Slide 20 - Website and Live Demonstration

FireWatch website:

- URL: `http://127.0.0.1:5000`
- Live wall: camera stream and current status
- Cameras: add webcam, IP camera, or RTSP source
- Classifier: upload and inspect RGB/thermal evidence
- Alerts: review stored alert events
- History: inspect previous evidence and decisions

Live behavior:

- RGB model detects visible fire/smoke and draws boxes.
- Thermal model only returns a fire/no-fire score.
- FireWatch uses temporal confirmation before raising a live alert.
- A paired thermal camera should represent the same scene as the RGB camera.
- The operator remains responsible for confirming an incident.

---

## Slide 21 - Conclusion and Future Scope

### Conclusion

The project delivers an AI-assisted forest-camera monitoring prototype combining RGB object detection, thermal classification, capture pairing, temporal confirmation, evidence storage, and a live web dashboard. The completed RGB run demonstrates useful fire/smoke detection performance, with stronger smoke than fire results. The thermal model provides a separate classification channel but requires calibration because of a cold-frame false positive.

### Future scope

- Add small-flame and matchstick-scale hard examples only if that operating condition is required.
- Add reliable thermal event metadata and clean grouped splits.
- Calibrate RGB and thermal scores on validation data.
- Add image registration and timestamp tolerance checks for camera pairs.
- Train and validate a dedicated NIR model only after obtaining labels.
- Evaluate on a locked, leakage-audited test set.
- Add operator-reviewed SMS/voice notification integration.
- Add model version display and automatic checkpoint reload in the dashboard.

---

## Slide 22 - Co-guide Meeting Proof

Insert the approved proof here:

- Meeting date: `[DATE]`
- Meeting mode: `[IN PERSON / ONLINE]`
- Co-guide: `[NAME AND DESIGNATION]`
- Discussion points: dataset preparation, model training, thermal classification, live demo, and evaluation plan
- Proof image or signed confirmation: `[INSERT IMAGE OR APPROVAL]`

Do not claim this slide is complete until the actual proof is inserted.

---

## Slide 23 - SDG Goals

### SDG 11 - Sustainable Cities and Communities

The project supports safer communities by providing earlier detection of fire and smoke hazards around forest edges, public infrastructure, and monitored camera zones.

### SDG 13 - Climate Action

Early fire detection can support faster response, reduce fire spread, and help limit damage to forests, ecosystems, and carbon stores.

### SDG 15 - Life on Land

Forest-fire monitoring supports protection of terrestrial ecosystems, wildlife habitat, and biodiversity.

**Recommended mapping:** SDG 13 and SDG 15 are the strongest direct matches. SDG 11 is a supporting application-impact goal.

---

## Slide 24 - Project Stakeholders and Existing Application Gap

### Client

- Academic project guide and department
- Institution requiring a working AI and web demonstration
- Potential forest, campus, industrial, or surveillance deployment partner

### Customer / end user

- Forest and wildlife monitoring teams
- Industrial safety operators
- Campus or facility security staff
- Emergency-response coordinators
- Researchers demonstrating multimodal fire detection

### User needs

- Early warning of visible fire or smoke
- Fewer nuisance alerts
- Evidence that can be reviewed after an alert
- Live status for multiple camera sources
- A clear distinction between detection, review, and confirmed fire

### Existing application gap

| Existing solutions | Our project |
|---|---|
| Manual or single-camera monitoring | Live dashboard with multiple camera records |
| RGB-only visual evidence | RGB detection plus separate thermal classification |
| One-frame alerts | Temporal confirmation and cooldown rules |
| Unexplained alerts | Boxes, scores, capture IDs, snapshots, and review status |
| Unrelated multimodal uploads can be fused | Paired RGB/thermal uploads require a shared capture ID |

**Key differentiator:** A human-reviewed FireWatch workflow combines visible fire/smoke localization with same-instance thermal verification and auditable evidence instead of treating a single confidence score as a final emergency decision.

---

## Slide 25 - References

Use the following references as the initial bibliography and verify publisher details, volume, issue, pages, and DOI before final submission.

1. B. U. Toreyin, Y. Dedeoglu, U. Gudukbay, and A. E. Cetin, “Computer vision based method for real-time fire and flame detection,” *Pattern Recognition Letters*, 2006.
2. T. Celik, H. Ozkaramanli, and H. Demirel, “Fire and smoke detection without sensors: Image processing based approach,” *Proceedings of the 15th European Signal Processing Conference*, 2007.
3. M. Frizzi, M. Bacciu, A. Angeletti, M. Prati, G. S. Tascini, and F. S. Lombardi, “Convolutional neural network for video fire and smoke detection,” *IEEE International Conference on Image Processing / related conference publication*, 2016.
4. K. Muhammad, J. Ahmad, Z. Lv, P. Bellinato, J. J. P. C. Rodrigues, and S. W. Baik, “Efficient deep CNN-based fire detection and localization in video surveillance applications,” *IEEE Transactions on Systems, Man, and Cybernetics: Systems*, 2018.
5. A. Krizhevsky, I. Sutskever, and G. E. Hinton, “ImageNet classification with deep convolutional neural networks,” *Advances in Neural Information Processing Systems*, 2012.
6. Ultralytics, “YOLO11 Documentation,” Ultralytics official documentation, accessed 2026.
7. Ultralytics, “Ultralytics Python Usage and Train Mode Documentation,” official documentation, accessed 2026.
8. Project implementation files: `train_forest_fire_detector.py`, `train_thermal_classifier.py`, `infer_video.py`, `firewatch/core/detector.py`, and `webapp/app.py`.
9. Project artifacts: `PROJECT_PROGRESS_REPORT.md`, `CLASSIFICATION_METRICS.md`, `MULTIMODAL_WEB_ARCHITECTURE.md`, and leakage-audit reports under `project_data/artifacts/leakage_audit/`.

---

## Slide 26 - Thank You and Queries

# Thank You

## Questions?

Contact: `[STUDENT EMAIL]`

Project: Forest Fire Detection and Multimodal Alert System
