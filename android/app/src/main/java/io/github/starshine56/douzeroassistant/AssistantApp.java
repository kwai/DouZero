package io.github.starshine56.douzeroassistant;

import android.app.Application;
import android.util.Log;
import org.opencv.android.OpenCVLoader;

public final class AssistantApp extends Application {
    @Override public void onCreate() {
        super.onCreate();
        boolean loaded = OpenCVLoader.initLocal();
        Log.i("DouZeroAssistant", "OpenCV local init=" + loaded);
        AssistantRuntime.initialize(this, loaded);
    }
}
