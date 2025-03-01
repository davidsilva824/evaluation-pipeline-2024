# models in which it works:

import subprocess

# Define the model path
MODEL_PATH = "babylm/ltgbert-100m-2024"
MODEL_BASENAME = MODEL_PATH.split("/")[-1]

def run_evaluation():
    command = [
        "python", "-m", "lm_eval",
        "--model", "hf-mlm",  # Ensure MLM is used
        "--model_args", f"pretrained={MODEL_PATH},backend=mlm,trust_remote_code=True",  # Explicitly set MLM class
        "--tasks", "blimp_irregular_past_participle_verbs,blimp_irregular_plural_subject_verb_agreement_1,blimp_irregular_plural_subject_verb_agreement_2,blimp_wh_island,blimp_adjunct_island,blimp_complex_NP_island,blimp_sentential_subject_island,blimp_regular_plural_subject_verb_agreement_1,blimp_regular_plural_subject_verb_agreement_2",
        "--device", "cuda:1",
        "--batch_size", "1",
        "--log_samples",
        "--output_path", f"results/blimp/{MODEL_BASENAME}/blimp_results.json"
    ]
    subprocess.run(command, check=True)

if __name__ == "__main__":
    run_evaluation()