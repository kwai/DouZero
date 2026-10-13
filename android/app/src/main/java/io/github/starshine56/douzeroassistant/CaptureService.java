package io.github.starshine56.douzeroassistant;

import android.app.*;
import android.content.Intent;
import android.graphics.PixelFormat;
import android.hardware.display.DisplayManager;
import android.media.ImageReader;
import android.media.projection.MediaProjection;
import android.media.projection.MediaProjectionManager;
import android.os.Handler;
import android.os.IBinder;
import android.os.Looper;

public final class CaptureService extends Service {
    static final String EXTRA_RESULT_CODE = "capture.resultCode";
    static final String EXTRA_RESULT_DATA = "capture.resultData";
    private static final String CHANNEL = "screen-analysis";
    private MediaProjection projection;
    private ImageReader reader;

    @Override public void onCreate() {
        super.onCreate();
        NotificationManager nm = getSystemService(NotificationManager.class);
        nm.createNotificationChannel(new NotificationChannel(CHANNEL, "屏幕识牌", NotificationManager.IMPORTANCE_LOW));
        Notification notification = new Notification.Builder(this, CHANNEL)
                .setSmallIcon(android.R.drawable.ic_menu_view)
                .setContentTitle("斗地主 AI 助手")
                .setContentText("正在本地分析屏幕；点击系统停止共享可立即结束")
                .setOngoing(true).build();
        startForeground(1, notification);
    }

    @Override public int onStartCommand(Intent intent, int flags, int startId) {
        int resultCode = intent.getIntExtra(EXTRA_RESULT_CODE, Activity.RESULT_CANCELED);
        Intent resultData = intent.getParcelableExtra(EXTRA_RESULT_DATA);
        if (resultCode != Activity.RESULT_OK || resultData == null) { stopSelf(); return START_NOT_STICKY; }
        MediaProjectionManager manager = getSystemService(MediaProjectionManager.class);
        projection = manager.getMediaProjection(resultCode, resultData);
        projection.registerCallback(new MediaProjection.Callback() {
            @Override public void onStop() { stopSelf(); }
        }, new Handler(Looper.getMainLooper()));
        int width = getResources().getDisplayMetrics().widthPixels;
        int height = getResources().getDisplayMetrics().heightPixels;
        int density = getResources().getDisplayMetrics().densityDpi;
        reader = ImageReader.newInstance(width, height, PixelFormat.RGBA_8888, 2);
        projection.createVirtualDisplay("DouZeroAssistant", width, height, density,
                DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR, reader.getSurface(), null, null);
        reader.setOnImageAvailableListener(r -> FramePipeline.offerLatest(r), null);
        return START_NOT_STICKY;
    }

    @Override public void onDestroy() {
        if (reader != null) reader.close();
        if (projection != null) projection.stop();
        super.onDestroy();
    }

    @Override public IBinder onBind(Intent intent) { return null; }
}
