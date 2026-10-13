plugins { id("com.android.application") }

android {
    namespace = "io.github.starshine56.douzeroassistant"
    compileSdk = 35

    defaultConfig {
        applicationId = "io.github.starshine56.douzeroassistant"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0-dev"
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
}
