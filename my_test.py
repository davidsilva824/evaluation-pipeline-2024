from surprisal import AutoHuggingFaceModel

sentences = [
    "This moster is a rat eater",
    "This moster is a rats eater",
    "This moster is a mice eater",
    "This moster is a rat catcher",
    "This moster is a rats catcher",
    "This moster is a mice catcher",
    "This moster is a duck feeder",
    "This moster is a ducks feeder"
    "This moster is a geese feeder",
    "This moster is a goose feeder",
    "This moster is a hand catcher",
    "This moster is a hands catcher",
    "This moster is a foot catcher",
    "This moster is a feet catcher",
]

m = AutoHuggingFaceModel.from_pretrained('phonemetransformers/GPT2-85M-BPE-TXT')
m.to('cuda') # optionally move your model to GPU!

for result in m.surprise(sentences):
    print(result)



