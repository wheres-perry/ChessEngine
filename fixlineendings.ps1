# Convert line endings from DOS (CRLF) to Unix (LF) for specified file types
# Recursively processes all files starting from current working directory

Write-Host "Converting line endings from DOS to Unix (recursive from $(Get-Location))..." -ForegroundColor Green

# Define file extensions to process
$extensions = @('*.py', '*.md', '*.toml', '*.sh')

# Get all matching files recursively
$files = Get-ChildItem -Path . -Recurse -Include $extensions -File

if ($files.Count -eq 0) {
    Write-Host "No files found with extensions: $($extensions -join ', ')" -ForegroundColor Yellow
    exit
}

Write-Host "Found $($files.Count) files to process..." -ForegroundColor Cyan

# Process each file
foreach ($file in $files) {
    try {
        Write-Host "Processing: $($file.FullName)" -ForegroundColor Gray
        
        # Read file content
        $content = Get-Content $file.FullName -Raw
        
        if ($content) {
            # Convert CRLF to LF
            $content = $content -replace "`r`n", "`n"
            # Convert any remaining CR to LF
            $content = $content -replace "`r", "`n"
            
            # Write back with UTF8 encoding (no BOM)
            $utf8NoBom = New-Object System.Text.UTF8Encoding $false
            [System.IO.File]::WriteAllText($file.FullName, $content, $utf8NoBom)
        }
    }
    catch {
        Write-Host "Error processing $($file.FullName): $($_.Exception.Message)" -ForegroundColor Red
    }
}

Write-Host "Line ending conversion completed!" -ForegroundColor Green
Write-Host "Processed files in $(Get-Location) and all subdirectories." -ForegroundColor Green

# Pause to see results
Read-Host "Press Enter to continue"