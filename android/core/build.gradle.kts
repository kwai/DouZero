import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins { id("org.jetbrains.kotlin.jvm") }

java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}
kotlin { compilerOptions { jvmTarget.set(JvmTarget.JVM_17) } }

dependencies {
    compileOnly("com.microsoft.onnxruntime:onnxruntime:1.22.0")
    testImplementation(kotlin("test"))
    testImplementation("com.microsoft.onnxruntime:onnxruntime:1.22.0")
    testImplementation("org.json:json:20240303")
}
tasks.test { useJUnitPlatform() }
