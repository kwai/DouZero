package io.github.starshine56.douzeroassistant;

import android.app.Service;
import android.content.Intent;
import android.graphics.Color;
import android.graphics.PixelFormat;
import android.graphics.drawable.GradientDrawable;
import android.os.Handler;
import android.os.IBinder;
import android.os.Looper;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.WindowManager;
import android.widget.LinearLayout;
import android.widget.TextView;

/** Read-only overlay. It never receives AccessibilityService privileges or sends touch events. */
public final class OverlayService extends Service {
    private WindowManager manager;
    private LinearLayout panel;
    private WindowManager.LayoutParams params;
    private final Handler main = new Handler(Looper.getMainLooper());
    private TextView title;
    private TextView recommendation;
    private TextView candidates;
    private TextView counter;
    private final Runnable refresh = new Runnable() {
        @Override public void run() {
            AssistantSnapshot s = AssistantRuntime.current();
            title.setText("✥  " + s.status() + (s.hand().isEmpty() ? "" : " · 手牌 " + s.hand()));
            recommendation.setText("推荐：" + s.recommendation());
            candidates.setText("Top 3：" + s.candidates() + "\n上家：" + empty(s.leftPlay()) + "  下家：" + empty(s.rightPlay()) + "  底牌：" + empty(s.bottom()));
            counter.setText("剩余牌：\n" + s.remaining());
            main.postDelayed(this, 350);
        }
    };

    private static String empty(String value) { return value.isEmpty() ? "--" : value; }

    @Override public void onCreate() {
        super.onCreate();
        manager = getSystemService(WindowManager.class);
        title = text("✥  AI 等待稳定牌局", 14, 0xffffd54f);
        recommendation = text("推荐：--", 16, 0xff81c784);
        candidates = text("Top 3：--", 11, Color.LTGRAY);
        counter = text("剩余牌：等待识别", 11, 0xff90caf9);
        panel = new LinearLayout(this);
        panel.setOrientation(LinearLayout.VERTICAL);
        panel.setPadding(14, 8, 14, 8);
        GradientDrawable background = new GradientDrawable();
        background.setColor(0xdd141414);
        background.setCornerRadius(18);
        panel.setBackground(background);
        panel.addView(title); panel.addView(recommendation); panel.addView(candidates); panel.addView(counter);
        params = new WindowManager.LayoutParams(WindowManager.LayoutParams.WRAP_CONTENT,
                WindowManager.LayoutParams.WRAP_CONTENT, WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
                WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE | WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN,
                PixelFormat.TRANSLUCENT);
        params.gravity = Gravity.TOP | Gravity.START;
        params.x = getResources().getDisplayMetrics().widthPixels / 2;
        params.y = 0;
        final float[] down = new float[4];
        title.setOnTouchListener((v, e) -> {
            if (e.getActionMasked() == MotionEvent.ACTION_DOWN) {
                down[0] = e.getRawX(); down[1] = e.getRawY(); down[2] = params.x; down[3] = params.y;
            } else if (e.getActionMasked() == MotionEvent.ACTION_MOVE) {
                params.x = (int) (down[2] + e.getRawX() - down[0]);
                params.y = (int) (down[3] + e.getRawY() - down[1]);
                manager.updateViewLayout(panel, params);
            }
            return true;
        });
        manager.addView(panel, params);
        main.post(refresh);
    }

    private TextView text(String value, float size, int color) {
        TextView view = new TextView(this); view.setText(value); view.setTextSize(size); view.setTextColor(color); return view;
    }

    @Override public void onDestroy() {
        main.removeCallbacks(refresh);
        if (panel != null) manager.removeView(panel);
        super.onDestroy();
    }
    @Override public IBinder onBind(Intent intent) { return null; }
}
