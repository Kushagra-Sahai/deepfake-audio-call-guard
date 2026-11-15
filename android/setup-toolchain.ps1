# Installs a minimal Android build toolchain (no Android Studio) into C:\android:
#   portable JDK 17, Android command-line tools + SDK 35, Gradle 8.10.2.  ~1 GB download.
# Usage:  powershell -ExecutionPolicy Bypass -File android\setup-toolchain.ps1
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # makes Invoke-WebRequest much faster
$root = "C:\android"
New-Item -ItemType Directory -Force $root, "$root\dl" | Out-Null

function Get-Zip($url, $name, $dest) {
    $zip = "$root\dl\$name"
    if (-not (Test-Path $zip)) { Write-Host "downloading $name"; Invoke-WebRequest $url -OutFile $zip }
    if (-not (Test-Path $dest)) { Write-Host "unzipping $name"; Expand-Archive $zip -DestinationPath $dest }
}

# JDK 17 (Gradle 8.10 / AGP 8.7 do not run on very new JDKs)
Get-Zip "https://aka.ms/download-jdk/microsoft-jdk-17-windows-x64.zip" "jdk17.zip" "$root\jdk-tmp"
if (-not (Test-Path "$root\jdk17")) { Move-Item (Get-ChildItem "$root\jdk-tmp" | Select-Object -First 1).FullName "$root\jdk17" }

# Android command-line tools -> sdk\cmdline-tools\latest
Get-Zip "https://dl.google.com/android/repository/commandlinetools-win-11076708_latest.zip" "cmdline-tools.zip" "$root\clt-tmp"
if (-not (Test-Path "$root\sdk\cmdline-tools\latest")) {
    New-Item -ItemType Directory -Force "$root\sdk\cmdline-tools" | Out-Null
    Move-Item "$root\clt-tmp\cmdline-tools" "$root\sdk\cmdline-tools\latest"
}

# Gradle
Get-Zip "https://services.gradle.org/distributions/gradle-8.10.2-bin.zip" "gradle.zip" "$root\gradle"   # -> $root\gradle\gradle-8.10.2

$env:JAVA_HOME = "$root\jdk17"
$env:ANDROID_HOME = "$root\sdk"
$sdkm = "$root\sdk\cmdline-tools\latest\bin\sdkmanager.bat"
Write-Host "accepting SDK licenses + installing platform-tools, android-35, build-tools 35"
# sdkmanager ignores piped "y" answers from PowerShell; write the standard license hashes instead
New-Item -ItemType Directory -Force "$root\sdk\licenses" | Out-Null
Set-Content -Encoding ascii "$root\sdk\licenses\android-sdk-license" "`n24333f8a63b6825ea9c5514f83c2829b004d1fee`n8933bad161af4178b1185d1a37fbf41ea5269c55`nd56f5187479451eabf01fb78af6dfcb131a6481e"
Set-Content -Encoding ascii "$root\sdk\licenses\android-sdk-preview-license" "`n84831b9409646a918e30573bab4c9c91346d8abd"
cmd /c "$sdkm --sdk_root=$root\sdk platform-tools platforms;android-35 build-tools;35.0.0"

Write-Host "done:"
& "$root\jdk17\bin\java.exe" -version
& "$root\sdk\platform-tools\adb.exe" version
