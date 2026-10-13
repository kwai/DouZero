package io.github.starshine56.douzeroassistant;

/** Immutable result published from recognition to the read-only overlay. */
record AssistantSnapshot(String status, String hand, String bottom, String leftPlay,
                         String rightPlay, String recommendation, String candidates, String remaining) {
    static AssistantSnapshot initial() {
        return new AssistantSnapshot("等待屏幕画面", "", "", "", "", "--", "--", "等待稳定识别");
    }
}
