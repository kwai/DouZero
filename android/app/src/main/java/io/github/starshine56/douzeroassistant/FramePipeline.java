package io.github.starshine56.douzeroassistant;

import android.graphics.Bitmap;
import android.media.Image;
import android.media.ImageReader;
import android.os.Handler;
import android.os.HandlerThread;
import java.nio.ByteBuffer;
import java.util.concurrent.atomic.AtomicBoolean;

/** Latest-frame-only conversion boundary. Processing is off the capture callback thread. */
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
        int width = image.getWidth();
        int height = image.getHeight();
        int rowStride = plane.getRowStride();
        int pixelStride = plane.getPixelStride();
        if (pixelStride != 4) { image.close(); BUSY.set(false); return; }
        ByteBuffer source = plane.getBuffer();
        byte[] copy = new byte[source.remaining()];
        source.get(copy);
        image.close();
        HANDLER.post(() -> {
            Bitmap padded = null;
            Bitmap cropped = null;
            try {
                int paddedWidth = rowStride / pixelStride;
                padded = Bitmap.createBitmap(paddedWidth, height, Bitmap.Config.ARGB_8888);
                padded.copyPixelsFromBuffer(ByteBuffer.wrap(copy));
                cropped = paddedWidth == width ? padded : Bitmap.createBitmap(padded, 0, 0, width, height);
                if (cropped != padded) padded.recycle();
                AssistantRuntime.process(cropped);
                cropped = null; // runtime owns and recycles it
            } catch (Throwable t) {
                if (cropped != null) cropped.recycle();
                if (padded != null && !padded.isRecycled()) padded.recycle();
            } finally {
                BUSY.set(false);
            }
        });
    }
}
