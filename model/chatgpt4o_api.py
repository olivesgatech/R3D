from PIL import Image
import torchvision.transforms as transforms
import base64
from openai import OpenAI
import io
import re

client = OpenAI(api_key="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")
#client = OpenAI(api_key="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")

def tensor_to_base64_images(image_tensors):
    """
    Convert a batched tensor of images into a list of Base64-encoded strings.
    
    Args:
        image_tensors (torch.Tensor): A tensor containing stacked image data (B, C, H, W).
    
    Returns:
        list: List of Base64-encoded image strings.
    """
    to_pil = transforms.ToPILImage()
    base64_images = []

    for tensor in image_tensors:
        # Convert tensor to PIL image
        pil_image = to_pil(tensor)
        
        # Save the PIL image to a BytesIO buffer
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG")
        
        # Base64 encode the image
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        base64_images.append(base64_image)

    return base64_images

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def tensor_to_pil_images(image_tensors):
    """
    Convert a batched tensor of images into a list of PIL images.
    Args:
        image_tensors (torch.Tensor): A tensor containing stacked image data (B, C, H, W).
    Returns:
        list: List of PIL images.
    """
    to_pil = transforms.ToPILImage()
    pil_images = [to_pil(tensor).tobytes() for tensor in image_tensors]  # Convert each tensor to PIL image
    return pil_images


def prepare_message_with_images(video_frames, human_prompt):
    """
    Prepare a message with text and multiple images for API submission.

    Args:
        video_frames (list): List of PIL.Image objects representing video frames.
        human_prompt (str): User prompt text.

    Returns:
        dict: Formatted messages for the OpenAI API.
    """
    # Prepare the text part of the message
    messages = [
        {"role": "system", "content": "You are a helpful assistant trained to predict fine-grained labels."},
        {"role": "user", "content": human_prompt}
    ]

    images = tensor_to_base64_images(video_frames)
    
    # images = []
    # f = open('/home/seulgi/work/darai-anticipation/FUTR_proposed/image_path.txt', 'r')

    # for x in f.readlines():
    #     images.append(encode_image(x.replace('\n', '')))

    # f.close()

    # Encode each image in video_frames and add to messages
    for base64_image in images:
        messages.append({
            "role": "user",
            "content": [{
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            }]
        })

    return messages

def prompt_post_processing(response, time):
    """
    Extract fine-grained labels from the API response text by finding the sequence of numbers
    following the "Answer:" section.
    Args:
        response (str): The response text from the OpenAI API.
    Returns:
        list: A list of fine-grained labels as integers.
    """
    # Use regex to find the "Answer:" section and extract numbers.
    match_answer = re.search(r"Answer:\s*([\d,.\s]+)", response)
    match_number_seq = re.findall(r"(\d+(?:,\s*\d+)*)", response)
    if match_answer:
        labels_text = match_answer.group(1).strip()
        labels_text.replace('.', ',')
        labels_answer = [int(label.strip()) for label in labels_text.split(",") if label.strip().isdigit()]
        labels_answer = [min(label, 47) for label in labels_answer]
        if len(labels_answer) == time:
            return labels_answer
        
    if match_number_seq:
        labels_text = max(match_number_seq, key=lambda x: len(x.replace('.', ',').split(",")))
        labels_max_num = [int(label.strip()) for label in labels_text.split(",")]
        labels_max_num = [min(label, 47) for label in labels_max_num]
        if len(labels_max_num) == time:
            return labels_max_num
        

    if match_answer and match_number_seq:
        return labels_answer if len(labels_answer) >= len(labels_max_num) else labels_max_num
    elif match_answer and not match_number_seq:
        return labels_answer
    elif not match_answer and match_number_seq:
        return labels_max_num
    else:
        print("No valid label sequence found in the response.")
        return []

    
def get_fine_grained_labels(video_frames, human_prompt, time):
    """
    Generate fine-grained labels using OpenAI GPT-4 Turbo (Vision API).

    Args:
        video_frames (list): List of PIL.Image objects representing video frames.
        human_prompt (str): Textual prompt provided by the user.

    Returns:
        str: GPT-generated fine-grained labels.
    """
    message = prepare_message_with_images(video_frames, human_prompt)
    # Sending the image to the model
    response = client.chat.completions.create(
        model="chatgpt-4o-latest",
        #model="gpt-4o-mini",
        #model="gpt-4o",
        messages=message,
        #max_tokens=1000,
    )
    print(response.choices[0].message.content)

    # Print the response
    return response.choices[0].message.content, prompt_post_processing(response.choices[0].message.content, time)