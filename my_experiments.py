from transformers import AutoModelForMaskedLM

MODEL_PATH = "SzegedAI/babylm24_MLSM_strict"

# This forces a fresh download of the model
model = AutoModelForMaskedLM.from_pretrained(MODEL_PATH, trust_remote_code=True, force_download=True)