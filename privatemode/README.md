## Prequisites

The following services need to be started for the script to work:

```
docker run -p 8080:8080 ghcr.io/edgelesssys/privatemode/privatemode-proxy:latest --apiKey <>
```

Download the binary from [here](https://github.com/qdrant/qdrant/releases) and run:

```
./qdrant
```

The API for OCR of the PDF document:

```
docker run -p 8000:8000 -d --rm robwilkes/unstructured-api
```

The local embedding model is optional depending on your setup:

```
docker run --platform linux/amd64 -p 9090:80 --pull always ghcr.io/huggingface/text-embeddings-inference:cpu-1.6 --model-id intfloat/e5-base-v2
```

Lastly, you can run the script with an [uv](https://github.com/astral-sh/uv) installation:

```
NVIDIA_API_KEY=<key> PM_API_KEY=<key> uv run main.py
```

Alternatively use the requirements.txt in the root to install the dependencies.


## Kubernetes deployment

### Helm chart for the embedding model

Following the instructions of:
https://artifacthub.io/packages/helm/infracloud-charts/text-embeddings-inference

```bash
helm install tei infracloud-charts/text-embeddings-inference -n as --create-namespace -f embedding.yaml
```

Then update the LB IP in the script.

### Helm chart for the Unstructed API

Following the instructions of:
https://github.com/kkacsh321/unstructured-api-helm-chart


```bash
helm install unstructured-api unstructured-api/unstructured-api -n as --create-namespace -f unstructured.yaml
```

Then update the LB IP in the script.