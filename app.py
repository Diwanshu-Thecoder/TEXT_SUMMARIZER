from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re 
from fastapi.templating import Jinja2Templates # UI
from fastapi.responses import HTMLResponse
from pathlib import Path

app = FastAPI(title="Text Summarizer App", description="Text Summarization using T5", version="1.0")

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_summary_model"

model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)

# device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)
model.eval()

templates = Jinja2Templates(directory=str(BASE_DIR))

class DialogueInput(BaseModel):
    dialogue: str

def clean_data(text):
    text = re.sub(r"\r\n", " ", text) # lines
    text = re.sub(r"\s+", " ", text) # spaces
    text = re.sub(r"<.*?>", " ", text) # html tags <p> <h1>
    text = text.strip().lower()
    return text

def summarize_dialogue(dialogue : str) -> str:
    dialogue = clean_data(dialogue) # clean
    if not dialogue:
        raise ValueError("Dialogue must not be empty")

    # tokenize
    inputs = tokenizer(
        f"summarize: {dialogue}",
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    # generate the summary => token ids
    with torch.inference_mode():
        targets = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=150,
            num_beams=4,
            early_stopping=True
        )
    
    # decoded our output
    summary = tokenizer.decode(targets[0], skip_special_tokens=True) # EOS, SEP
    return summary


# API endpoints
@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    if not dialogue_input.dialogue.strip():
        raise HTTPException(status_code=422, detail="Dialogue must not be empty")
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})