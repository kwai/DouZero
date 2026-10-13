# Android automatic-recognition assistant

This branch adds a clean-room, local-first assistant around Apache-2.0 DouZero.

## Product boundary

- Standard three-player DouDizhu, no wild cards in the first release.
- Android landscape capture through MediaProjection.
- Read-only overlay with remaining-card counts and Top-3 recommendations.
- No accessibility service, simulated touch, account access, SMS/location/device-ID collection, remote screenshots, or cleartext traffic.
- Recognition uncertainty or violation of the 54-card invariant blocks recommendations and asks for correction.

## Architecture

1. `CaptureService` receives explicit system screen-sharing consent.
2. `FramePipeline` keeps only the latest frame to avoid analyzing animations late.
3. A clean-room NCNN detector will emit rank, confidence and normalized bounds.
4. ROI routing separates hand, center play, left play, right play, bottom cards and remaining counters.
5. `MultiFrameStabilizer` requires repeated observations.
6. `RoundState` enforces physical card counts and maintains unseen cards.
7. The inference adapter converts a verified state to the existing DouZero environment.
8. A read-only overlay presents Top-3 actions, score and concise reasons.

## Reference policy

- `kwai/DouZero`: source dependency under Apache-2.0; existing LICENSE retained.
- Other public repositories without an explicit license and the supplied proprietary APK are behavior/architecture references only. Their source, model weights, artwork, protocol and server endpoints are not copied.

## Current status

Implemented: card normalization, 54-card invariant, round observations, uncertainty gate, multi-frame voting, minimum-permission Android shell, MediaProjection foreground service, latest-frame backpressure, and CI.

Not yet implemented: trained detector weights, ROI calibration UI, native NCNN bridge, full legal-play validation, DouZero mobile model conversion, overlay recommendation UI, and on-device validation.
