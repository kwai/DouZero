package io.github.starshine56.douzeroassistant;

import android.media.Image;
import android.media.ImageReader;
import java.util.concurrent.atomic.AtomicBoolean;

/** Back-pressure boundary for the future NCNN detector. Never queues stale animation frames. */
final class FramePipeline {
    private static final AtomicBoolean BUSY = new AtomicBoolean(false);
    private FramePipeline() {}

    static void offerLatest(ImageReader reader) {
        Image image = reader.acquireLatestImage();
        if (image == null) return;
        if (!BUSY.compareAndSet(false, true)) { image.close(); return; }
        try {
            // TODO: crop calibrated ROIs and invoke the clean-room NCNN JNI detector.
            // The image is deliberately not persisted or uploaded.
        } finally {
            image.close();
            BUSY.set(false);
        }
    }
}
