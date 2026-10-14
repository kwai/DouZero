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
import android.view.View;
import android.view.WindowManager;
import android.widget.LinearLayout;
import android.widget.TextView;

/** Two draggable read-only overlays: concise AI hint + independent card counter. */
public final class OverlayService extends Service {
    private WindowManager wm;
    private LinearLayout aiPanel, details, recorderPanel;
    private WindowManager.LayoutParams aiParams, recorderParams;
    private final Handler main = new Handler(Looper.getMainLooper());
    private TextView headline, recommendation, alternatives, detailToggle, hide, detailText, recorderText;
    private boolean collapsed, detailsShown;

    private final Runnable refresh = new Runnable() {
        @Override public void run() {
            AssistantSnapshot s = AssistantRuntime.current();
            headline.setText(collapsed ? "AI" : shortStatus(s.status()));
            recommendation.setText("建议：" + empty(s.recommendation()));
            alternatives.setText("备选：" + shortCandidates(s.candidates()));
            detailText.setText("手牌：" + empty(s.hand()) + "\n上家：" + empty(s.leftPlay()) +
                    "  下家：" + empty(s.rightPlay()) + "  底牌：" + empty(s.bottom()) +
                    "\n评分：" + empty(s.candidates()));
            recorderText.setText("记牌器\n" + s.remaining());
            main.postDelayed(this, 350);
        }
    };

    @Override public void onCreate() {
        super.onCreate();
        wm = getSystemService(WindowManager.class);
        headline = text("识别中", 14, 0xffffd54f);
        recommendation = text("建议：--", 19, 0xff81c784);
        alternatives = text("备选：--", 12, Color.LTGRAY);
        detailToggle = text("详情", 12, Color.WHITE);
        hide = text("隐藏", 12, Color.WHITE);
        detailToggle.setPadding(16, 5, 8, 5); hide.setPadding(16, 5, 8, 5);
        detailText = text("", 11, Color.LTGRAY);
        details = new LinearLayout(this); details.setOrientation(LinearLayout.VERTICAL);
        details.setVisibility(View.GONE); details.addView(detailText);

        LinearLayout header = new LinearLayout(this); header.setGravity(Gravity.CENTER_VERTICAL);
        header.addView(headline); header.addView(detailToggle); header.addView(hide);
        aiPanel = panel(); aiPanel.addView(header); aiPanel.addView(recommendation); aiPanel.addView(alternatives); aiPanel.addView(details);
        aiParams = params(8, 8);
        bindDrag(headline, aiPanel, aiParams, true);
        detailToggle.setOnClickListener(v -> setDetails(!detailsShown));
        hide.setOnClickListener(v -> setCollapsed(true));
        wm.addView(aiPanel, aiParams);

        recorderText = text("记牌器\n等待牌局", 11, 0xff90caf9);
        recorderPanel = panel(); recorderPanel.addView(recorderText);
        recorderParams = params(8, 220);
        bindDrag(recorderText, recorderPanel, recorderParams, false);
        wm.addView(recorderPanel, recorderParams);
        main.post(refresh);
    }

    private String shortStatus(String raw) {
        if (raw == null) return "识别中";
        if (raw.contains("轮到你")) return "轮到你";
        if (raw.contains("等待上家") || raw.contains("等待下家") || raw.contains("对局中")) return "等待对手";
        if (raw.contains("错误") || raw.contains("失败") || raw.contains("暂停")) return "⚠ " + raw.substring(0, Math.min(raw.length(), 24));
        if (raw.contains("新局") || raw.contains("完整发牌") || raw.contains("未开始")) return "等待新局";
        return "识别中";
    }

    private String shortCandidates(String raw) {
        if (raw == null || raw.isEmpty() || raw.equals("--")) return "--";
        String[] parts = raw.split("\\|");
        if (parts.length <= 1) return raw;
        StringBuilder out = new StringBuilder();
        for (int i = 1; i < Math.min(parts.length, 3); i++) {
            String move = parts[i].trim().split(" = ")[0];
            if (out.length() > 0) out.append(" / ");
            out.append(move.equals("Pass") ? "不出" : move);
        }
        return out.length() == 0 ? "--" : out.toString();
    }

    private void setDetails(boolean show) {
        detailsShown = show; details.setVisibility(show ? View.VISIBLE : View.GONE);
        detailToggle.setText(show ? "收起" : "详情"); wm.updateViewLayout(aiPanel, aiParams);
    }

    private void setCollapsed(boolean value) {
        collapsed = value;
        recommendation.setVisibility(value ? View.GONE : View.VISIBLE);
        alternatives.setVisibility(value ? View.GONE : View.VISIBLE);
        details.setVisibility(!value && detailsShown ? View.VISIBLE : View.GONE);
        detailToggle.setVisibility(value ? View.GONE : View.VISIBLE);
        hide.setVisibility(value ? View.GONE : View.VISIBLE);
        recorderPanel.setVisibility(value ? View.GONE : View.VISIBLE);
        aiPanel.setPadding(value ? 18 : 14, value ? 10 : 8, value ? 18 : 14, value ? 10 : 8);
        wm.updateViewLayout(aiPanel, aiParams);
    }

    private void bindDrag(View handle, View target, WindowManager.LayoutParams p, boolean restoreOnTap) {
        final float[] down = new float[4]; final boolean[] moved = new boolean[1];
        handle.setOnTouchListener((v, e) -> {
            if (e.getActionMasked() == MotionEvent.ACTION_DOWN) {
                moved[0] = false; down[0] = e.getRawX(); down[1] = e.getRawY(); down[2] = p.x; down[3] = p.y;
            } else if (e.getActionMasked() == MotionEvent.ACTION_MOVE) {
                float dx=e.getRawX()-down[0], dy=e.getRawY()-down[1]; if (Math.abs(dx)+Math.abs(dy)>8) moved[0]=true;
                p.x=(int)(down[2]+dx); p.y=(int)(down[3]+dy); wm.updateViewLayout(target,p);
            } else if (e.getActionMasked() == MotionEvent.ACTION_UP && restoreOnTap && collapsed && !moved[0]) setCollapsed(false);
            return true;
        });
    }

    private LinearLayout panel() {
        LinearLayout p=new LinearLayout(this); p.setOrientation(LinearLayout.VERTICAL); p.setPadding(14,8,14,8);
        GradientDrawable bg=new GradientDrawable(); bg.setColor(0xdd141414); bg.setCornerRadius(18); p.setBackground(bg); return p;
    }
    private WindowManager.LayoutParams params(int x,int y) {
        WindowManager.LayoutParams p=new WindowManager.LayoutParams(WindowManager.LayoutParams.WRAP_CONTENT,WindowManager.LayoutParams.WRAP_CONTENT,
                WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE|WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN,PixelFormat.TRANSLUCENT);
        p.gravity=Gravity.TOP|Gravity.START; p.x=x; p.y=y; return p;
    }
    private TextView text(String value,float size,int color) { TextView v=new TextView(this);v.setText(value);v.setTextSize(size);v.setTextColor(color);return v; }
    private static String empty(String value) { return value==null||value.isEmpty()?"--":value; }

    @Override public void onDestroy() {
        main.removeCallbacks(refresh);
        if(aiPanel!=null)runCatchingRemove(aiPanel); if(recorderPanel!=null)runCatchingRemove(recorderPanel);
        super.onDestroy();
    }
    private void runCatchingRemove(View v) { try { wm.removeView(v); } catch(Exception ignored) {} }
    @Override public IBinder onBind(Intent intent) { return null; }
}
