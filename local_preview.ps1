$ErrorActionPreference = "Stop"

# Simulate the repository name as 'slides' (GitHub Pages default path is /<repo-name>/)
$RepoName = "slides"
$OutputDir = "dist_local"
$TargetDir = "$OutputDir/$RepoName"

Write-Host "cleaning up..."
if (Test-Path $OutputDir) { Remove-Item -Recurse -Force $OutputDir }
New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null

Write-Host "Building Jackknife..."
Push-Location jackknife
# Ensure dependencies are installed
if (-not (Test-Path "node_modules")) { pnpm install }
pnpm build --base /$RepoName/jackknife/ --out ../$TargetDir/jackknife
Pop-Location

Write-Host "Building Water Wavelets..."
Push-Location water_wavelets
if (-not (Test-Path "node_modules")) { pnpm install }
pnpm build --base /$RepoName/water_wavelets/ --out ../$TargetDir/water_wavelets
Pop-Location

Write-Host "Copying index.html..."
Copy-Item index.html $TargetDir/index.html

Write-Host "`nBuild complete!"
Write-Host "Starting local server..."
Write-Host "Please open http://127.0.0.1:8080/$RepoName/ in your browser."

# Use npx to run http-server without installing it globally
npx http-server $OutputDir --cors -c-1
