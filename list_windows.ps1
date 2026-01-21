# Script to list all visible windows with their titles
Add-Type @"
using System;
using System.Text;
using System.Runtime.InteropServices;

public class Win32 {
  public delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);

  [DllImport("user32.dll")] public static extern bool EnumWindows(EnumWindowsProc lpEnumFunc, IntPtr lParam);
  [DllImport("user32.dll")] public static extern int GetWindowText(IntPtr hWnd, StringBuilder text, int count);
  [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr hWnd);
}
"@

$windows = @()

$callback = [Win32+EnumWindowsProc]{
  param($hWnd, $lParam)
  if ([Win32]::IsWindowVisible($hWnd)) {
    $sb = New-Object System.Text.StringBuilder 512
    [void][Win32]::GetWindowText($hWnd, $sb, $sb.Capacity)
    $title = $sb.ToString()
    if ($title -ne "") {
      $script:windows += $title
    }
  }
  return $true
}

[Win32]::EnumWindows($callback, [IntPtr]::Zero) | Out-Null

Write-Host "=== All Visible Windows ===" -ForegroundColor Green
Write-Host ""
$windows | Sort-Object | ForEach-Object {
  Write-Host $_
}
Write-Host ""
Write-Host "Total windows found: $($windows.Count)" -ForegroundColor Yellow
