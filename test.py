from google import genai
from google.genai import types

def generate():
    client = genai.Client(
        api_key="AIzaSyAn235cmhBr6wtQOVXWOWAw_pZdFCpP2SU"
    )

    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents="hi",
    )

    print(response.text)

if __name__ == "__main__":
    generate()
