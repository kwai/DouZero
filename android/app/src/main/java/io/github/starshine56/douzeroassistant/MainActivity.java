package io.github.starshine56.douzeroassistant;

import android.app.Activity;
import android.content.Intent;
import android.media.projection.MediaProjectionManager;
import android.os.Bundle;
import android.provider.Settings;
import android.net.Uri;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;

public final class MainActivity extends Activity {
    private static final int CAPTURE_REQUEST = 1001;
    private MediaProjectionManager projectionManager;

    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        projectionManager = getSystemService(MediaProjectionManager.class);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(48, 48, 48, 48);
        TextView status = new TextView(this);
        status.setText("版本 0.4.0-clean-ui（versionCode 8）\n\n默认仅显示状态、建议和备选；手牌与评分放入详情，完整记牌器使用独立可拖动悬浮块。\n仅本地分析，不上传截图，不自动点击游戏。");
        status.setTextSize(18);
        Button overlay = new Button(this);
        overlay.setText("授予悬浮窗权限");
        overlay.setOnClickListener(v -> startActivity(new Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                Uri.parse("package:" + getPackageName()))));
        Button start = new Button(this);
        start.setText("开始屏幕识别");
        start.setOnClickListener(v -> startActivityForResult(projectionManager.createScreenCaptureIntent(), CAPTURE_REQUEST));
        root.addView(status);
        root.addView(overlay);
        root.addView(start);
        setContentView(root);
    }

    @Override protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == CAPTURE_REQUEST && resultCode == RESULT_OK && data != null) {
            Intent service = new Intent(this, CaptureService.class)
                    .putExtra(CaptureService.EXTRA_RESULT_CODE, resultCode)
                    .putExtra(CaptureService.EXTRA_RESULT_DATA, data);
            startForegroundService(service);
            if (Settings.canDrawOverlays(this)) startService(new Intent(this, OverlayService.class));
            finish();
        }
    }
}
