import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins { id("org.jetbrains.kotlin.jvm") }
java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}
kotlin { compilerOptions { jvmTarget.set(JvmTarget.JVM_17) } }

dependencies {
    api(project(":core"))
    compileOnly("org.openpnp:opencv:4.9.0-0")
    testImplementation(kotlin("test"))
    testImplementation("org.openpnp:opencv:4.9.0-0")
}
tasks.test {
    useJUnitPlatform()
    systemProperty("templates.dir", file("../app/src/main/assets/templates").absolutePath)
}
