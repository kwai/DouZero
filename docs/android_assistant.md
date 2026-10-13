# Android automatic-recognition assistant

This branch adds a local-first assistant around Apache-2.0 DouZero. The Kotlin rule engine, observation encoder, ONNX adapter and parity tests are adapted from the Apache-2.0 project `Superbaaaaber/Auto-doudizhu-on-Android-based-on-Douzero`; see `NOTICE`.

## Product boundary

- Standard three-player DouDizhu, no wild cards in the first release.
- Android landscape capture through MediaProjection.
- Read-only overlay with remaining-card counts and Top-3 recommendations.
- No accessibility service, simulated touch, account access, SMS/location/device-ID collection, remote screenshots, or cleartext traffic.
- Recognition uncertainty or violation of the 54-card invariant blocks recommendations and asks for correction.

## Architecture

1. `CaptureService` receives explicit system screen-sharing consent.
2. `FramePipeline` keeps only the latest frame to avoid analyzing animations late.
3. The first recognition backend uses OpenCV rank-template matching; an optional NCNN detector can replace it after a licensed/locally-trained model is available.
4. Resolution-independent ROI routing separates hand, center play, left play, right play, bottom cards and remaining counters. A first 2400×1080 profile was measured from the supplied screenshot.
5. `MultiFrameStabilizer` requires repeated observations.
6. `RoundState` enforces physical card counts and maintains unseen cards.
7. The inference adapter converts a verified state to the existing DouZero environment.
8. A read-only overlay presents Top-3 actions, score and concise reasons.

## Reference policy

- `kwai/DouZero`: source dependency under Apache-2.0; existing LICENSE retained.
- `Superbaaaaber/Auto-doudizhu-on-Android-based-on-Douzero`: Apache-2.0 Kotlin rules/encoder/ONNX adapter/parity tests, adapted with attribution and without its automatic-control layer.
- Other public repositories without an explicit license and the supplied proprietary APK are behavior/architecture references only. Their source, model weights, artwork, protocol and server endpoints are not copied.

## Current status

Implemented: card normalization, 54-card invariant, round observations, uncertainty gate, multi-frame voting, minimum-permission Android shell, MediaProjection foreground service, latest-frame backpressure, Kotlin legal-move engine, DouZero observation encoding, ONNX Runtime adapter, Top-3 decision logic, a draggable read-only overlay, a screenshot-specific ROI profile, OpenCV bitmap conversion and live hand/bottom/left/right recognition, bundled Apache-2.0 mobile templates, overlay live recognition updates, real-screenshot regression tests, parity/safety tests, and CI.

Not yet implemented: bundled ONNX weights, templates calibrated specifically for the supplied independent-app skin, calibration UI, robust turn/pass event reduction into `GameState`, live AI decisions, and on-device validation. The current build can display recognition attempts but must not yet be described as a reliable automatic recommender.
