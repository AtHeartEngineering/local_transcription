# local_transcription

## Settings

Make sure to set the IP, port, and HuggingFace API key in the `settings.toml` file. The huggingface API key is used to fetch the models, everything else is ran locally.

```toml
[host]
ip="192.168.0.99"
port=5063

[settings]
upload_folder="uploads"
model_folder="models"
hf_token="HUGGINGFACE API KEY"
```

## Usage

`python host.py` on the server

`python client_upload.py audio.mp3` on the client