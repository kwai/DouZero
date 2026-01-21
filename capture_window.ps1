param(
  [Parameter(Mandatory=$true)]
  [string]$window_name,

  [Parameter(Mandatory=$true)]
  [int]$interval,

  [Parameter(Mandatory=$true)]
  [string]$file_path,

  [Parameter(Mandatory=$false)]
  [switch]$exact_match,

  [Parameter(Mandatory=$false)]
  [switch]$trace_titles
)

# Load required assemblies
Add-Type -AssemblyName System.Drawing

# Define Win32 API functions
if (-not ('Win32Capture3' -as [type])) {
Add-Type @"
using System;
using System.Text;
using System.Runtime.InteropServices;

public class Win32Capture3 {
  public delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);

  [DllImport("user32.dll")] public static extern bool EnumWindows(EnumWindowsProc lpEnumFunc, IntPtr lParam);
  [DllImport("user32.dll")] public static extern int GetWindowText(IntPtr hWnd, StringBuilder text, int count);
  [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr hWnd);
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr hWnd, out RECT rect);
  [DllImport("user32.dll")] public static extern bool GetClientRect(IntPtr hWnd, out RECT rect);
  [DllImport("user32.dll")] public static extern bool ClientToScreen(IntPtr hWnd, ref POINT pt);
  [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint processId);
  [DllImport("shcore.dll")] public static extern int SetProcessDpiAwareness(int awareness);

  [StructLayout(LayoutKind.Sequential)]
  public struct RECT { public int Left; public int Top; public int Right; public int Bottom; }

  [StructLayout(LayoutKind.Sequential)]
  public struct POINT { public int X; public int Y; }
}
"@
}

try {
  # 2 = PROCESS_PER_MONITOR_DPI_AWARE
  [void][Win32Capture3]::SetProcessDpiAwareness(2)
} catch {
  # Ignore if the call is unavailable on this system
}

# Create output directory if it doesn't exist
New-Item -ItemType Directory -Force -Path $file_path | Out-Null

# Function to find window by title with exact or partial matching
function Get-WindowHandleByTitle($title, $exactMatch = $false) {
  $script:found = [IntPtr]::Zero
  $script:matchedTitle = ""

  $callback = [Win32Capture3+EnumWindowsProc]{
    param($hWnd, $lParam)
    if (-not [Win32Capture3]::IsWindowVisible($hWnd)) { return $true }

    $sb = New-Object System.Text.StringBuilder 512
    [void][Win32Capture3]::GetWindowText($hWnd, $sb, $sb.Capacity)
    $windowTitle = $sb.ToString()

    if ($trace_titles) {
      Write-Host "DEBUG: Checking window: '$windowTitle'" -ForegroundColor Gray
    }

    $isMatch = $false
    if ($exactMatch) {
      $isMatch = ($windowTitle -eq $title)
    } else {
      $isMatch = ($windowTitle -like "*$title*")
    }

    if ($isMatch) {
      $script:found = $hWnd
      $script:matchedTitle = $windowTitle
      if ($trace_titles) {
        Write-Host "DEBUG: MATCH FOUND! '$windowTitle'" -ForegroundColor Green
      }
      return $false
    }
    return $true
  }

  [Win32Capture3]::EnumWindows($callback, [IntPtr]::Zero) | Out-Null

  if ($script:found -ne [IntPtr]::Zero -and $script:matchedTitle -ne "") {
    Write-Host "Found window: '$script:matchedTitle'" -ForegroundColor Green
  }

  return $script:found
}

# Function to find window by process name
function Get-WindowByProcessName($processName) {
  $processes = Get-Process | Where-Object { $_.ProcessName -like "*$processName*" }

  foreach ($proc in $processes) {
    if ($proc.MainWindowHandle -ne [IntPtr]::Zero) {
      $sb = New-Object System.Text.StringBuilder 512
      [void][Win32Capture3]::GetWindowText($proc.MainWindowHandle, $sb, $sb.Capacity)
      $title = $sb.ToString()
      Write-Host "Found window via process: '$title' (Process: $($proc.ProcessName))" -ForegroundColor Green
      return $proc.MainWindowHandle
    }
  }
  return [IntPtr]::Zero
}

Write-Host "Starting screenshot capture..." -ForegroundColor Cyan
Write-Host "Window: $window_name"
Write-Host "Interval: $interval seconds"
Write-Host "Save path: $file_path"
Write-Host "Exact match: $exact_match"
if ($trace_titles) {
  Write-Host "Trace mode: ON" -ForegroundColor Yellow
}
Write-Host "Press Ctrl+C to stop"
Write-Host ""

# Try to find the window first before starting the loop
Write-Host "Searching for window..." -ForegroundColor Yellow
$testHandle = Get-WindowHandleByTitle $window_name $exact_match

if ($testHandle -eq [IntPtr]::Zero) {
  Write-Host "Window not found with title matching. Trying process name search..." -ForegroundColor Yellow
  $testHandle = Get-WindowByProcessName $window_name

  if ($testHandle -eq [IntPtr]::Zero) {
    Write-Host ""
    Write-Host "ERROR: Could not find window!" -ForegroundColor Red
    Write-Host "Suggestions:" -ForegroundColor Yellow
    Write-Host "1. Run .\list_windows.ps1 to see all window titles"
    Write-Host "2. Try with -exact_match flag: -exact_match"
    Write-Host "3. Try with -trace_titles flag to see all windows being checked: -trace_titles"
    Write-Host "4. Make sure the window is visible and not minimized"
    exit 1
  }
}

Write-Host "Starting capture loop..." -ForegroundColor Cyan
Write-Host ""

$captureCount = 0

# Main capture loop
while ($true) {
  $hwnd = Get-WindowHandleByTitle $window_name $exact_match

  if ($hwnd -ne [IntPtr]::Zero) {
    $clientRect = New-Object Win32Capture3+RECT
    if ([Win32Capture3]::GetClientRect($hwnd, [ref]$clientRect)) {
      $pt = New-Object Win32Capture3+POINT
      $pt.X = $clientRect.Left
      $pt.Y = $clientRect.Top
      [void][Win32Capture3]::ClientToScreen($hwnd, [ref]$pt)
      $width = $clientRect.Right - $clientRect.Left
      $height = $clientRect.Bottom - $clientRect.Top

      if ($width -gt 0 -and $height -gt 0) {
        $bmp = New-Object System.Drawing.Bitmap $width, $height
        $gfx = [System.Drawing.Graphics]::FromImage($bmp)
        $gfx.CopyFromScreen($pt.X, $pt.Y, 0, 0, $bmp.Size)

        $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $filename = "screenshot_$timestamp.png"
        $savePath = Join-Path $file_path $filename

        $bmp.Save($savePath, [System.Drawing.Imaging.ImageFormat]::Png)
        $gfx.Dispose()
        $bmp.Dispose()

        $captureCount++
        Write-Host "$(Get-Date -Format 'HH:mm:ss') - Captured #$captureCount : $filename" -ForegroundColor Green
      }
    }
  } else {
    Write-Host "$(Get-Date -Format 'HH:mm:ss') - Window not found: $window_name" -ForegroundColor Red
  }

  Start-Sleep -Seconds $interval
}
