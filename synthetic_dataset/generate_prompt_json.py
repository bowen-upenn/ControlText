import json
import cv2
import base64
import requests
import numpy as np
from tqdm import tqdm


def query_openai_gpt_4v(image_path, api_key):
    # we have to crop the image before converting it to base64
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    _, buffer = cv2.imencode('.jpg', image)
    image_bytes = np.array(buffer).tobytes()
    base64_image = base64.b64encode(image_bytes).decode('utf-8')

    messages = "Your task is to generate a detailed prompt for ControlNet to replicate the specific font style seen in an image. " \
               "Focus on describing the unique visual characteristics of the font that make it different from other fonts, " \
               "such as letter shape, line weight, glyph width, and any distinct features and styles of the typeface that stand out. " \
               "Avoid mentioning general attributes and ignore any shape distortion, perspective transformation, or rotation in the text. " \
               "Provide the description in two sentences. If there are multiple lines, describe each line separately. " \
               "Here is the prompt:"
    max_tokens = 400

    prompt = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": messages},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        "max_tokens": max_tokens
    }

    # Send request to OpenAI API
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=prompt)
    response_json = response.json()

    # Process the response
    # Check if the response is valid and contains the expected data
    if 'choices' in response_json and len(response_json['choices']) > 0:
        completion_text = response_json['choices'][0].get('message', {}).get('content', '')
    else:
        completion_text = ""

    return completion_text


if __name__ == "__main__":
    # Load the API key
    with open("openai_key.txt", "r") as api_key_file:
        api_key = api_key_file.read()

    # Generate the data
    data = []
    error_count = 0
    for i in tqdm(range(300000)):
        text_idx = i // 3
        font_idx = i % 3
        image_idx = str(text_idx) + '_' + str(font_idx)

        try:
            with open("test_dataset/texts/" + image_idx + ".txt", 'r') as file:
                texts = file.read()
        except:
            error_count += 1
            continue

        source_image_path = f"test_dataset/target_curved/{image_idx}.png"
        target_image_path = f"test_dataset/target/{image_idx}.png"
        # print('image_idx', image_idx, 'source_image_path', source_image_path, 'target_image_path', target_image_path, 'texts', texts)

        data.append({"source": source_image_path, "target": target_image_path,
                     "prompt": f'A black background with the texts "{texts}" that have no shape distortion, curvature, or rotation. Follow the same fonts in the condition image.'})

        # font_description = query_openai_gpt_4v(source_image_path, api_key)
        # print('Font description:', font_description)

        # data.append({"source": source_image_path, "target": target_image_path,
        #              "prompt": f'A black background with the texts "{texts}" that have no shape distortion, curvature, or rotation.' + font_description})

    print('Error count:', error_count, 'number of data:', len(data))

    # Write to a JSON file
    with open('prompt.json', 'w') as file:
        for entry in data:
            json.dump(entry, file)
            file.write('\n')
