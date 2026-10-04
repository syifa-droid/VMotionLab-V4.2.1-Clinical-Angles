$ErrorActionPreference = "Stop"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git is not installed. Install Git for Windows first."
}

git init
git branch -M main
git add .
git commit -m "Initial release: VMotionLab V4.2.1 Clinical Angles"

Write-Host ""
Write-Host "Local repository created." -ForegroundColor Green
Write-Host "Create an empty GitHub repository named:" -ForegroundColor Cyan
Write-Host "VMotionLab-V4.2.1-Clinical-Angles"
Write-Host ""
Write-Host "Then run:"
Write-Host 'git remote add origin https://github.com/YOUR_USERNAME/VMotionLab-V4.2.1-Clinical-Angles.git'
Write-Host 'git push -u origin main'
