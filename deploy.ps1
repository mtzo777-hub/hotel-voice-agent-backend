param(
  [string]$TAG = "manual-" + (Get-Date -Format "yyyyMMdd-HHmmss")
)

$ErrorActionPreference = "Stop"

$PROJECT="hotel-voice-agent-487409"
$REGION="asia-southeast1"
$REPO="hotel-voice-agent-repo"
$IMAGE="$REGION-docker.pkg.dev/$PROJECT/$REPO/hotel-voice-agent-backend:$TAG"
$SERVICE="hotel-voice-agent-backend"

Write-Host "== Building image: $IMAGE =="

# Make sure you're authenticated to Artifact Registry
gcloud config set project $PROJECT | Out-Null
gcloud auth print-access-token | podman login -u oauth2accesstoken --password-stdin "https://$REGION-docker.pkg.dev" | Out-Null

podman build -t "hotel-voice-agent-backend:$TAG" .
podman tag "hotel-voice-agent-backend:$TAG" $IMAGE
podman push $IMAGE

Write-Host "== Deploying to Cloud Run =="
gcloud run deploy $SERVICE `
  --image $IMAGE `
  --region $REGION `
  --platform managed `
  --allow-unauthenticated `
  --port 8080 `
  --timeout 300

Write-Host "DONE. Deployed tag: $TAG"
