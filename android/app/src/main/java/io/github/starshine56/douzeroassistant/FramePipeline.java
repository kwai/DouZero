package io.github.starshine56.douzeroassistant;

import android.media.Image;
import android.media.ImageReader;
import android.os.Handler;
import android.os.HandlerThread;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.imgproc.Imgproc;
import java.nio.ByteBuffer;
import java.util.concurrent.atomic.AtomicBoolean;

/** Latest-frame-only ImageReader RGBA -> OpenCV BGR conversion. */
final class FramePipeline {
    private static final AtomicBoolean BUSY = new AtomicBoolean(false);
    private static final HandlerThread THREAD = new HandlerThread("card-recognition");
    private static final Handler HANDLER;
    static { THREAD.start(); HANDLER = new Handler(THREAD.getLooper()); }
    private FramePipeline() {}

    static void offerLatest(ImageReader reader) {
        Image image = reader.acquireLatestImage();
        if (image == null) return;
        if (!BUSY.compareAndSet(false, true)) { image.close(); return; }
        Image.Plane plane = image.getPlanes()[0];
        int width = image.getWidth(), height = image.getHeight();
        int rowStride = plane.getRowStride(), pixelStride = plane.getPixelStride();
        if (pixelStride != 4 || rowStride < width * 4) { image.close(); BUSY.set(false); return; }
        ByteBuffer source = plane.getBuffer(); source.rewind();
        byte[] copy = new byte[rowStride * height];
        source.get(copy, 0, Math.min(source.remaining(), copy.length));
        image.close();
        HANDLER.post(() -> {
            Mat padded = new Mat(height, rowStride / 4, CvType.CV_8UC4);
            Mat rgba = null; Mat bgr = new Mat();
            try {
                padded.put(0, 0, copy);
                rgba = padded.submat(new Rect(0, 0, width, height));
                Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR);
                AssistantRuntime.processBgr(bgr);
            } catch (Throwable ignored) {
            } finally {
                if (rgba != null) rgba.release();
                padded.release(); bgr.release(); BUSY.set(false);
            }
        });
    }
}
