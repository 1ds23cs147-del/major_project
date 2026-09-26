# Forest Fire Detection Project Progress Report

Date: 2026-09-23

## 1. Purpose

This report records the work completed from the document `Forest_Fire_Model_Training_and_Data_Leakage_Guide.docx`, the evidence collected, the current training and audit state, the measured results, unresolved blockers, and the next execution steps.

The main objective is to produce a defensible forest-fire detection project using:

- RGB fire/smoke object detection
- Thermal fire/no-fire classification
- Leakage-free train/validation/test evaluation
- Validation-only fusion tuning
- A working FireWatch web application

## 2. Guide Requirements

The co-guide document requires:

1. Audit RGB and thermal data separately.
2. Find exact duplicates within training and validation.
3. Find exact duplicates across training and validation.
4. Find perceptual near-duplicates using Hamming distance threshold 4.
5. Review flagged near-duplicate pairs manually.
6. Keep source video/event/camera groups together where possible.
7. Apply augmentation only after the clean split.
8. Track RGB precision, recall, F1, mAP@50, and mAP@50-95.
9. Re-evaluate the thermal 100% result on clean data.
10. Perform hard-example mining using existing data.
11. Use group-aware cross-validation when practical.
12. Tune one major training variable at a time.
13. Tune RGB/thermal fusion weights only on validation data.
14. Keep the final test set locked.
15. Retrain and compare the cleaned model against the baseline.

## 3. Dataset Inventory

### RGB detector dataset

Location:

`project_data/datasets/rgb_thermal/fasdd_yolo_real`

Manifest:

`project_data/datasets/rgb_thermal/fasdd_yolo_real/data.yaml`

Classes:

- fire
- smoke

Approximate image counts:

- Train: 47,660 images
- Validation: 31,770 images
- Test: 15,884 images
- Train plus validation audited: 79,430 images

### Thermal classifier dataset

Location:

`project_data/datasets/rgb_thermal/flame3_cls_real/thermal`

Manifest:

`project_data/datasets/rgb_thermal/flame3_cls_real/thermal/data.yaml`

Classes:

- fire
- no_fire

Image counts:

- Train fire: 497
- Train no_fire: 92
- Validation fire: 125
- Validation no_fire: 24
- Total train: 589
- Total validation: 149

Separate thermal test data exists under:

`project_data/datasets/rgb_thermal/flame3_cls_test/thermal`

It contains five fire images and five no-fire images.

Important limitation: thermal filenames are sequential numbers such as `00001.JPG` and `00498.JPG`. They do not identify a source video, event, camera, or recording session.

## 4. Training State

The previous RGB detector run was:

`project_data/runs/forest_fire/fasdd_real_smoke_30ep`

Checkpoint files:

- `project_data/runs/forest_fire/fasdd_real_smoke_30ep/weights/epoch9.pt`
- `project_data/runs/forest_fire/fasdd_real_smoke_30ep/weights/last.pt`
- `models/rgb_fire_smoke_best.pt`

The run was resumed from `last.pt` and reached at least epoch 15 before it was stopped.

Last observed RGB detector metrics before stopping:

- Precision: 67.62%
- Recall: 57.26%
- mAP@50: 62.85%
- mAP@50-95: 37.43%

The previous run must be treated as a baseline only because its train/validation split was not yet proven to be free from leakage.

The existing thermal run was:

`project_data/runs/forest_fire/thermal_fire`

It ended at epoch 18. Its validation log was effectively perfect, but the result is not accepted as production evidence because the application verification found a severe cold-frame false positive.

Training status:

- Original RGB baseline training: completed 30/30 epochs from the resumed `last.pt` checkpoint
- Final checkpoint: `models/rgb_fire_smoke_best.pt`
- Thermal classifier checkpoint: `models/thermal_fire_no_fire_best.pt`
- Final validation metrics: precision 68.7%, recall 57.6%, mAP@50 63.7%, mAP@50-95 38.6%
- Class metrics: fire mAP@50 58.0%, smoke mAP@50 69.3%
- New clean-split training: not started
- The DOCX audit/clean-split gate is not being used as a prerequisite for this resumed baseline run, per the current project decision

### Current Thermal and Demo Pairing Design

- Thermal remains a classification-only model (`fire` / `no_fire`); it is not used for bounding-box detection.
- Paired RGB/thermal uploads now require an explicit shared `capture_id`.
- Two verified RGB/thermal demo pairs are documented in `webapp/demo_pairs/README.md`.
- A capture ID provides traceability, but the files must still be selected from the same physical capture; it cannot prove that unrelated files depict the same moment.
- FireWatch now exposes three verified RGB/thermal pairs through a one-click classifier catalog at `/api/demo-pairs`; selecting a pair fills both original classifier inputs and runs Analyse.

## 5. Audit Tooling Added

### Main audit script

`audit_dataset_leakage.py`

Capabilities:

- SHA-256 exact hashing
- Perceptual hashing with `imagehash`
- Hamming-distance threshold 4
- Separate RGB and thermal processing
- Train-within-split analysis
- Validation-within-split analysis
- Train-validation analysis
- Excel workbook output
- Source-group overlap reporting

### Optimized RGB near-duplicate audit

`audit_rgb_near_duplicates.py`

This uses a BK-tree for perceptual-hash queries instead of enumerating every train/validation pair. It is intended for the large 79,430-image RGB collection.

### Exact RGB quarantine script

`quarantine_rgb_exact_duplicates.py`

This creates a list of validation images that exactly duplicate training images. It does not delete or modify the original dataset.

### Dependencies added

The following dependencies were installed in the project virtual environment and added to `requirements.txt`:

- imagehash>=4.3.2
- openpyxl>=3.1.5
- pandas>=3.0.0

## 6. Audit Results

### Thermal audit

Report:

`project_data/artifacts/leakage_audit/thermal_duplicate_and_leakage_report.xlsx`

Results:

- Train exact duplicates: 0
- Validation exact duplicates: 0
- Cross train-validation exact duplicates: 0
- Train near-duplicate candidates: 86
- Validation near-duplicate candidates: 3
- Cross train-validation near-duplicate candidates: 37
- Group overlap records: 622

Interpretation:

The thermal data has no exact file duplicates, but it has 37 cross-split near-duplicate candidates. These require visual review and classification as one of:

- CONFIRMED_LEAKAGE
- VALID_SIMILAR_SCENE
- NEEDS_REVIEW

Because the thermal filenames contain no reliable event identifier, a defensible grouped split cannot yet be created automatically.

### RGB exact audit

Report:

`project_data/artifacts/leakage_audit/rgb_exact_group_audit.csv`

Results:

- Images scanned: 79,430
- Within-split exact duplicate groups: 78
- Cross-split exact duplicate hash groups: 36
- Validation images selected for quarantine: 66

Quarantine list:

`project_data/artifacts/leakage_audit/rgb_exact_cross_split_validation_quarantine.csv`

Interpretation:

The current RGB validation set is contaminated by exact copies of training images. Existing RGB validation metrics cannot be treated as clean final metrics.

### RGB inferred grouping

The filename prefixes were tested as possible source groups. They resolve mainly to:

- `fire`
- `smoke`
- `bothfireandsmoke`
- `neitherfirenorsmoke`

All of these appear in both train and validation. They are class prefixes, not recording-event groups, so they must not be used as a fake grouped split.

### RGB perceptual audit

The optimized BK-tree near-duplicate worker was launched after the exact audit. Its output is expected at:

`project_data/artifacts/leakage_audit/rgb_near_duplicate_review.xlsx`

Its progress log is:

`project_data/artifacts/leakage_audit/rgb_near_audit.log`

The earlier all-pairs implementation was stopped because it reached the hashing stage but became impractical during the large comparison stage. The original dataset was not modified.

Final RGB perceptual audit result:

- Train images indexed: 47,660
- Validation images queried: 31,770
- Cross-split near-duplicate candidates at Hamming threshold 4: 208,666
- Audit errors: 0

This is a review queue, not automatic proof of leakage. Each candidate must be visually reviewed before rebuilding the split.

## 6.1 Clean Candidate Dataset Build

The first conservative candidate split was built without modifying the original datasets:

Manifest:

`project_data/artifacts/clean_candidate_datasets/clean_split_manifest.txt`

RGB candidate:

- Train: 47,660 images
- Validation: 22,454 images
- Test: 15,884 images
- Validation images excluded: 9,316
- Missing RGB labels: 0

Thermal candidate:

- Train: 589 images
- Validation: 130 images
- Validation images excluded: 19

The candidate contains none of the paths listed in the RGB exact cross-split quarantine CSV. This is a structural candidate check, not proof that all near-duplicate pairs are validly classified. The RGB and thermal near-duplicate review gates remain open, and no training has started from this candidate.

## 7. FireWatch Application Results

The FireWatch verification script was run from:

`firewatch/verify.py`

Passed checks include:

- Templates render correctly
- Empty live wall renders correctly
- Camera page renders correctly
- Review page renders correctly
- History page renders correctly
- HTML tags are balanced
- Element IDs are unique
- JavaScript IDs resolve correctly
- CSS custom properties resolve
- RGB backend handles test images
- Blank RGB scene scores near zero
- Thermal hotspot scores high
- Fusion rules behave as expected
- Thermal veto behavior works
- Repeated-frame alert logic works
- Alert cooldown works
- Camera helper URL parsing works
- SQLite store round trips work
- Routes used by the UI exist

Known failing check:

- `cold frame scores ~0 on thermal`
- Observed thermal score: 0.90
- Expected thermal score: below 0.20

This confirms that the existing thermal model overpredicts fire on a cold frame and needs clean-data evaluation and retraining or calibration.

## 8. Checklist Status

### Dataset audit

- RGB Train Exact: completed by SHA-256 audit; 78 within-split duplicate groups found
- RGB Train Near: candidate report generated; within-split review not complete
- RGB Validation Exact: completed by SHA-256 audit
- RGB Validation Near: candidate report generated; within-split review not complete
- RGB Train-Validation Exact is zero in the original split: NOT COMPLETE; 36 cross-split hash groups affecting 66 validation files were found. The candidate excludes all listed RGB exact-quarantine validation paths, but a full content-hash audit of the candidate remains pending.
- RGB Train-Validation Near: candidate report generated with 208,666 candidates; review not complete
- Thermal Train Exact: completed; zero exact duplicate groups
- Thermal Train Near: completed; 86 candidates require review
- Thermal Validation Exact: completed; zero exact duplicate groups
- Thermal Validation Near: completed; 3 candidates require review
- Thermal Train-Validation Exact is zero: completed; zero exact duplicate groups
- Thermal Train-Validation Near: completed; 37 candidates require review

### Training and evaluation

- Split grouped by video/event/camera: NOT COMPLETE; metadata is missing or unreliable
- Augmentation applied after splitting: not yet completed for a clean rebuilt split
- RGB recall tracked with precision: completed for the old baseline, but not on a clean split
- Thermal 100% result re-evaluated on clean split: NOT COMPLETE; current result rejected due to cold-frame failure
- Fusion weights tuned only on validation: NOT COMPLETE
- Final test set remains locked: test files have not been used to tune settings
- Hard-example mining: NOT COMPLETE
- Group-based cross-validation: NOT COMPLETE
- Clean RGB retraining: NOT STARTED
- Clean thermal retraining: NOT STARTED
- Locked-test final evaluation: NOT STARTED

## 9. Why Retraining Has Not Started

Starting training now would produce another model using a split with known RGB exact leakage and unresolved thermal near-duplicates. That would make the final metrics difficult to defend to a co-guide or evaluator.

The epoch-15 RGB checkpoint is preserved as a baseline. It is not deleted and is not being used as the final model.

## 10. Next Process

### Step 1: Finish the RGB near-duplicate workbook

Check:

`project_data/artifacts/leakage_audit/rgb_near_audit.log`

Then verify:

`project_data/artifacts/leakage_audit/rgb_near_duplicate_review.xlsx`

### Step 2: Review the flagged pairs

For every cross-split near-duplicate candidate:

1. Open both image paths.
2. Compare the images visually.
3. Inspect filenames and available source information.
4. Mark the pair as `CONFIRMED_LEAKAGE`, `VALID_SIMILAR_SCENE`, or `NEEDS_REVIEW`.
5. If confirmed as leakage, keep the complete source group in only one split.

The review must be done for both RGB and thermal reports.

### Step 3: Validate and, after review, rebuild the RGB clean split

At minimum:

- Keep the original dataset untouched.
- Remove exact duplicate validation images from the candidate validation split.
- Do not move images into both splits.
- Preserve the locked test set.
- Save a manifest of every moved or quarantined image.

Current candidate build is complete and structurally validated, but it is not yet the final clean split because the 208,666 RGB near-duplicate candidates still lack review labels.

A fully grouped RGB split still requires real event/video metadata. Class prefixes must not be used as event groups.

### Step 4: Resolve thermal grouping

Preferred options:

- Obtain the original source-video or event manifest.
- Obtain timestamps or camera-session metadata.
- Ask the dataset provider how sequential frame ranges map to recording sessions.
- If no metadata exists, document that grouping cannot be guaranteed and use conservative near-duplicate quarantine with a clear limitation.

### Step 5: Train clean baselines

Only after the previous gates:

RGB detector:

- Use the cleaned RGB train/validation split.
- Keep test locked.
- Track precision, recall, F1, mAP@50, and mAP@50-95.
- Prioritize recall without allowing a severe precision regression.

Thermal classifier:

- Use the cleaned thermal train/validation split.
- Add realistic thermal augmentation only after splitting.
- Track accuracy, precision, recall, F1, balanced accuracy, and confusion matrix.
- Specifically test cold frames and hot non-fire examples.

### Step 6: Hard-example mining

Run the clean baseline models over training and validation images. Save:

- False negatives
- False positives
- Low-confidence cases
- Image path
- Ground-truth label
- Predicted label
- Confidence
- Failure condition

Manually verify labels before changing sampling weights.

### Step 7: Validation-only fusion tuning

Compare these RGB/thermal weight pairs on validation only:

- 30/70
- 40/60
- 45/55
- 50/50
- 60/40
- 70/30

Tune alert thresholds on validation only. Do not use the locked test set for choosing weights or thresholds.

### Step 8: Locked test evaluation

After all settings are frozen, evaluate once on the test set and save:

- RGB precision, recall, F1, mAP@50, mAP@50-95
- Thermal accuracy, precision, recall, F1, balanced accuracy
- RGB-only binary alert metrics
- RGB plus thermal metrics
- False-alert rate
- False-negative rate
- Confusion matrices
- Inference latency

### Step 9: FireWatch integration

Before calling the project complete:

- Replace or calibrate the thermal model that fails the cold-frame check.
- Keep temporal consistency checks enabled.
- Keep thermal veto behavior enabled.
- Test live camera behavior with no camera, one camera, and multiple cameras.
- Verify alerts are stored with evidence and scores.

## 11. Current Decision

Training is intentionally paused.

The project is not yet at the final-model stage because the current datasets have unresolved leakage and grouping issues. The most important next action is to finish the RGB near-duplicate review and obtain thermal source-event metadata. Once those are available, clean retraining and final evaluation can proceed.

## 12. Files Created or Updated During This Work

- `audit_dataset_leakage.py`
- `audit_rgb_near_duplicates.py`
- `quarantine_rgb_exact_duplicates.py`
- `run_rgb_audit.py`
- `requirements.txt`
- `project_data/artifacts/leakage_audit/thermal_duplicate_and_leakage_report.xlsx`
- `project_data/artifacts/leakage_audit/rgb_exact_group_audit.csv`
- `project_data/artifacts/leakage_audit/rgb_exact_cross_split_validation_quarantine.csv`
- `project_data/artifacts/leakage_audit/AUDIT_STATUS.md`

## 13. Final Summary

The project application is functional, and the audit workflow is now implemented. The current detector baseline has useful performance, but it cannot be presented as final because RGB exact leakage was found and the thermal classifier fails a cold-frame sanity check.

The next defensible milestone is a clean, documented split followed by fresh RGB and thermal training, validation-only fusion tuning, and one locked-test evaluation.

## 14. Report Maintenance Rule

This report is the project progress log and must be updated whenever new work changes the project state. Each update should include:

- Date and time of the change
- Files added or modified
- Commands or validation checks run
- Training epoch, checkpoint, and metrics when training changes
- Audit counts and report locations when data auditing changes
- Checklist items completed or still blocked
- The next action required

No new model result should be treated as current unless it is recorded here with its checkpoint path, dataset split, configuration, and evaluation source.
