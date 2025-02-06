## Prequisites

```
docker run -p 8080:8080 ghcr.io/edgelesssys/privatemode/privatemode-proxy:v1.6.0 --apiKey <>
```

```
./qdrant
```

```
docker run --platform linux/amd64 -p 9090:80 --pull always ghcr.io/huggingface/text-embeddings-inference:cpu-1.6 --model-id intfloat/e5-base-v2
```

```
docker run -p 8000:8000 -d --rm robwilkes/unstructured-api
```

PM_API_KEY=<key> uv run qdrant.py

```
