package io.github.starshine56.douzeroassistant;

import android.graphics.Rect;
import org.json.JSONObject;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;

/** Resolution-independent ROI profile. Coordinates are scaled from its reference screenshot. */
final class RecognitionProfile {
    final int referenceWidth;
    final int referenceHeight;
    final Map<String, Rect> regions;

    private RecognitionProfile(int width, int height, Map<String, Rect> regions) {
        this.referenceWidth = width;
        this.referenceHeight = height;
        this.regions = regions;
    }

    static RecognitionProfile load(InputStream stream) throws Exception {
        byte[] bytes = stream.readAllBytes();
        JSONObject root = new JSONObject(new String(bytes, StandardCharsets.UTF_8));
        int width = root.getInt("referenceWidth");
        int height = root.getInt("referenceHeight");
        if (width <= 0 || height <= 0) throw new IllegalArgumentException("invalid reference size");
        Map<String, Rect> regions = new HashMap<>();
        JSONObject source = root.getJSONObject("regions");
        java.util.Iterator<String> names = source.keys();
        while (names.hasNext()) {
            String name = names.next();
            JSONObject r = source.getJSONObject(name);
            Rect rect = new Rect(r.getInt("x"), r.getInt("y"),
                    r.getInt("x") + r.getInt("width"), r.getInt("y") + r.getInt("height"));
            if (rect.left < 0 || rect.top < 0 || rect.right > width || rect.bottom > height || rect.isEmpty())
                throw new IllegalArgumentException("invalid region: " + name);
            regions.put(name, rect);
        }
        return new RecognitionProfile(width, height, Map.copyOf(regions));
    }

    Rect scale(String name, int screenWidth, int screenHeight) {
        Rect r = regions.get(name);
        if (r == null) throw new IllegalArgumentException("unknown region: " + name);
        float sx = screenWidth / (float) referenceWidth;
        float sy = screenHeight / (float) referenceHeight;
        return new Rect(Math.round(r.left * sx), Math.round(r.top * sy),
                Math.round(r.right * sx), Math.round(r.bottom * sy));
    }
}
