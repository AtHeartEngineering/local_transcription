from flask import Flask, request, jsonify
import whisperx
import gc
import torch
from werkzeug.utils import secure_filename
import os
import toml

settings = toml.load('settings.toml')

app = Flask(__name__)
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'm4a'}
app.config['UPLOAD_FOLDER'] = settings['settings']['upload_folder']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

device = "cuda"
batch_size = 16
language = "en"
compute_type = "float16"
model_dir = settings['settings']['model_folder']
HF_TOKEN = settings['settings']['hf_token']

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'file' not in request.files:
        return jsonify(error="No file part"), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify(error="No selected file"), 400
    if file and allowed_file(file.filename):
        if file.filename is not None:
          filename = secure_filename(file.filename)
          file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
          file.save(file_path)

          # Transcribe with Whisper
          model = whisperx.load_model("large-v2", device, language=language, compute_type=compute_type, download_root=model_dir)
          audio = whisperx.load_audio(file_path)
          result = model.transcribe(audio, batch_size=batch_size)
          print(result["segments"])  # before alignment

          # Align Whisper output
          model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
          result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
          print(result["segments"])  # after alignment

        # Assign speaker labels
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=device)
        diarize_segments = diarize_model(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)

        # Clean-up
        gc.collect()
        torch.cuda.empty_cache()
        del model, model_a, diarize_model

        os.remove(file_path)  # Remove the uploaded file after processing

        return jsonify(result["segments"])

    return jsonify(error="Invalid file type"), 400

if __name__ == '__main__':
    app.run(debug=True, host=settings['host']['ip'], port=settings['host']['port'])
