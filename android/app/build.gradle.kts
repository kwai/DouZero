plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "io.github.starshine56.douzeroassistant"
    compileSdk = 35

    defaultConfig {
        applicationId = "io.github.starshine56.douzeroassistant"
        minSdk = 26
        targetSdk = 35
        versionCode = 3
        versionName = "0.2.1-diagnostic"
        ndk { abiFilters += "arm64-v8a" }
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    packaging { resources.excludes += setOf("META-INF/*.kotlin_module", "META-INF/versions/**") }
}

dependencies {
    implementation(project(":core"))
    implementation(project(":vision"))
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.22.0")
    implementation("org.opencv:opencv:4.10.0")
}
