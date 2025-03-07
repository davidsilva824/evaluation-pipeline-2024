### not working
import subprocess
import torch

# List of model paths
MODEL_PATHS = ["SzegedAI/babylm24_MLSM_strict"]

def run_evaluation(model_path):
    MODEL_BASENAME = model_path.split("/")[-1]  # Extract model name

    # Check if GPU is available and set the device
    if torch.cuda.is_available():
        device = f"cuda:{torch.cuda.current_device()}"
    else:
        device = "cpu"

    command = [
        "python", "-m", "lm_eval",
        "--model", "hf-mlm",  # Ensure MLM is used
        "--model_args", f"pretrained={model_path},backend=mlm,trust_remote_code=True",  # Explicitly set MLM class
        "--tasks", "blimp_irregular_past_participle_verbs,blimp_irregular_plural_subject_verb_agreement_1,blimp_irregular_plural_subject_verb_agreement_2,blimp_wh_island,blimp_adjunct_island,blimp_complex_NP_island,blimp_sentential_subject_island,blimp_regular_plural_subject_verb_agreement_1,blimp_regular_plural_subject_verb_agreement_2",
        "--device", device,
        "--batch_size", "1",
        "--log_samples",
        "--output_path", f"results/blimp/{MODEL_BASENAME}/blimp_results.json"
    ]

    subprocess.run(command, check=True)

if __name__ == "__main__":
    for model_path in MODEL_PATHS:
        run_evaluation(model_path)


