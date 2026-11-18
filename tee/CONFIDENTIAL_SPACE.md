# Running the mock workload on Google Confidential Space (CPU only)

This is the command sequence used for the one live run in August 2026 (mock workload, no model, no GPU, `n2d-standard-4`, debug image). It costs cents. You need a Google Cloud project with billing and the Confidential Computing API enabled. Run from the artifact root.

Pitfalls found in that run:

1. Build for `linux/amd64`. An arm64 image fails on the VM with `exec format error`.
2. The Dockerfile must `EXPOSE 7401`, or the launcher does not publish the port.
3. The workload accepts exactly one connection. Any reachability probe (`nc`, `curl`) uses it up. Restart the VM instead.
4. Use `stop` and `start`, not `reset`, to pick up a new image.

```bash
PROJECT_ID="<your project>"
REGION=us-central1; ZONE=us-central1-a
gcloud config set project "$PROJECT_ID"
gcloud services enable compute.googleapis.com confidentialcomputing.googleapis.com \
  artifactregistry.googleapis.com iamcredentials.googleapis.com

# only if the project has no VPC network
gcloud compute networks create ct-net --subnet-mode=custom
gcloud compute networks subnets create ct-subnet --network=ct-net --region=$REGION --range=10.0.0.0/24

gcloud artifacts repositories create ct --repository-format=docker --location=$REGION
gcloud auth configure-docker $REGION-docker.pkg.dev
docker buildx build --platform linux/amd64 -f deploy/Dockerfile \
  -t $REGION-docker.pkg.dev/$PROJECT_ID/ct/workload:mock --push .

gcloud compute instances create ct-workload \
  --zone=$ZONE --machine-type=n2d-standard-4 --network=ct-net --subnet=ct-subnet \
  --confidential-compute-type=SEV --maintenance-policy=TERMINATE \
  --shielded-secure-boot --scopes=cloud-platform --tags=ct-workload \
  --image-family=confidential-space-debug --image-project=confidential-space-images \
  --metadata=tee-image-reference=$REGION-docker.pkg.dev/$PROJECT_ID/ct/workload:mock,tee-container-log-redirect=true
gcloud compute firewall-rules create allow-ct-workload --network=ct-net \
  --allow=tcp:7401 --target-tags=ct-workload --source-ranges=<your IP>/32

# the workload prints {"verify_key": ..., "token": ...} on boot (debug image only)
gcloud compute instances get-serial-port-output ct-workload --zone=$ZONE \
  | grep -o '{"verify_key".*}' > /tmp/token.json

python -m evals_runner protocol-integrity-exp gcp-pin \
  --verify-key "$(python -c "import json;print(json.load(open('/tmp/token.json'))['verify_key'])")" \
  --token "$(python -c "import json;print(json.load(open('/tmp/token.json'))['token'])")"
IP=$(gcloud compute instances describe ct-workload --zone=$ZONE \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)')
python -m evals_runner protocol-integrity-exp spec --case mock --n 8 --spec /tmp/spec.json
python -m evals_runner protocol-integrity-exp certifier --case mock --spec /tmp/spec.json \
  --peer-host "$IP" --peer-port 7401 --attestation gcp --out /tmp/gcp_session

gcloud compute instances delete ct-workload --zone=$ZONE    # stop billing
```

A `VALID` certificate means the certifier verified a real Confidential Space token before trusting the workload's signature. It does not say anything about a GPU, and the debug image is not a production policy.
