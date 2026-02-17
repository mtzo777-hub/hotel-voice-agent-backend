param(
  [string]$TAG = "manual-" + (Get-Date -Format "yyyyMMdd-HHmmss")
)

$ErrorActionPreference = "Stop"

# Always run from this script's folder (so Dockerfile/context is correct)
Set-Location -Path $PSScriptRoot

$PROJECT = "hotel-voice-agent-487409"
$REGION  = "asia-southeast1"
$REPO    = "hotel-voice-agent-repo"
$SERVICE = "hotel-voice-agent-backend"

$LOCAL_IMAGE = "hotel-voice-agent-backend:$TAG"
$REMOTE_IMAGE = "$REGION-docker.pkg.dev/$PROJECT/$REPO/hotel-voice-agent-backend:$TAG"

Write-Host "== Context directory: $PSScriptRoot =="
Write-Host "== Building image: $REMOTE_IMAGE =="

# Set gcloud project (this may print an "environment tag" advisory warning; that's OK)
try {
  gcloud config set project $PROJECT | Out-Null
} catch {
  Write-Warning "gcloud config set project returned an error. If this is only the environment-tag advisory, it can be ignored. Details: $($_.Exception.Message)"
}

# Login to Artifact Registry for Podman push
gcloud auth print-access-token `
  | podman login -u oauth2accesstoken --password-stdin "https://$REGION-docker.pkg.dev" `
  | Out-Null

# Build + tag + push
podman build -t $LOCAL_IMAGE .
podman tag $LOCAL_IMAGE $REMOTE_IMAGE
podman push $REMOTE_IMAGE

Write-Host "== Deploying to Cloud Run =="

gcloud run deploy $SERVICE `
  --image $REMOTE_IMAGE `
  --region $REGION `
  --platform managed `
  --allow-unauthenticated `
  --port 8080 `
  --timeout 300

Write-Host "DONE. Deployed tag: $TAG"
Write-Host "Image: $REMOTE_IMAGE"
