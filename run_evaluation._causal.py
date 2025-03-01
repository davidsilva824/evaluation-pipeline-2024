import subprocess
# models in which it works:
# phonemetransformers/GPT2-85M-BPE-TXT (babble)
# bbunzeck/gpt-wee-small-curriculum
# bbunzeck/gpt-wee-medium-curriculum


# Define the model path at the beginning
MODEL_PATH = "bbunzeck/gpt-wee-medium-curriculum"
MODEL_BASENAME = MODEL_PATH.split("/")[-1]  # Extract model name

def run_evaluation():
    command = [
        "python", "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={MODEL_PATH},backend=causal,trust_remote_code=True",
        "--tasks", "blimp_irregular_past_participle_verbs,blimp_irregular_plural_subject_verb_agreement_1,blimp_irregular_plural_subject_verb_agreement_2,blimp_wh_island,blimp_adjunct_island,blimp_complex_NP_island,blimp_sentential_subject_island,blimp_regular_plural_subject_verb_agreement_1,blimp_regular_plural_subject_verb_agreement_2",
        "--device", "cuda:1",
        "--batch_size", "1",
        "--log_samples",
        "--output_path", f"results/blimp/{MODEL_BASENAME}/blimp_results.json"
    ]

    subprocess.run(command, check=True)

if __name__ == "__main__":
    run_evaluation()

