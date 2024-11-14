from byaldi import RAGMultiModalModel
import os
from google.colab import userdata
from pdf2image import convert_from_path
import base64
from io import BytesIO
import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor
from transformers import BitsAndBytesConfig
from PIL import Image
import requests
from pdf2image import convert_from_path


model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16)

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,quantization_config=bnb_config,device_map="auto")


userdata.get('HF_TOKEN')
os.environ["GROQ_API_KEY"] = userdata.get('GROQ_API_KEY')
model=RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2")
model.index(input_path="/content/RAINBOW.pdf",index_name="RAINBOW")


query="What is the ALOS (Rs.) as of Q1FY25, Occupancy (%) as of Q1FY25,ARPOB (Rs.) as of Q1FY25 of Rainbow Hospital"
result=model.search(query=query,k=1)
image_index = result[0]["page_num"] - 1
images = convert_from_path("/content/RAINBOW.pdf")
images[image_index]

processor = AutoProcessor.from_pretrained(model_id)
images[image_index].save('image.jpg', 'JPEG')
image = Image.open("/content/image.jpg")

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "Extract the following information and answer in bullet point: f'{query}'"}
    ]}
]

input_text = processor.apply_chat_template(
    messages, add_generation_prompt=True,
)
inputs = processor(
    image, input_text, return_tensors="pt"
).to(model.device)

output = model.generate(**inputs, max_new_tokens=120)

print(processor.decode(output[0][inputs["input_ids"].shape[-1]:]))
