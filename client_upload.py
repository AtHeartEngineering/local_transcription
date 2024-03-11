import requests
import sys
import toml

settings = toml.load('settings.toml')

def upload_audio_for_transcription(api_url, file_path):
    """
    Uploads an audio file to the transcription API and returns the response.

    Parameters:
    - api_url: The URL of the transcription API endpoint.
    - file_path: The path to the audio file to be uploaded.

    Returns:
    - A dictionary with the transcription result or error message.
    """
    files = {'file': open(file_path, 'rb')}
    response = requests.post(api_url, files=files)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Failed to transcribe audio. Status code: {response.status_code}, Message: {response.text}"}

# Example usage
api_url = f"http://{settings['host']['ip']}:{settings['host']['port']}/transcribe"
file_path = sys.argv[1]
result = upload_audio_for_transcription(api_url, file_path)
print(result)
