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
