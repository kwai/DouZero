# DouZero Assistant Android shell

Development build only. It currently requests screen sharing and establishes an in-memory frame pipeline. It does **not** yet recognize cards or produce recommendations.

Build with JDK 17, Android SDK 35, and Gradle 8.9:

```sh
gradle :app:assembleDebug
```

The manifest intentionally omits INTERNET and AccessibilityService.
