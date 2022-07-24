from transformers import Trainer, TrainingArguments

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from getsampletxt import get_sample_txt
import torch
import torch.nn.functional as F


txt = get_sample_txt(True)

# print(txt)
# classifier = pipeline("sentiment-analysis")

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

demo_num = 5
if demo_num == 1:
    classifier = pipeline("sentiment-analysis", model = model, tokenizer = tokenizer)
    res = classifier(txt)
    print(res)
if demo_num == 2:
    txt = "hi. How are you. I like machine learning"
    res = tokenizer(txt)
    print("tokenizer output", res)
    tokens = tokenizer.tokenize(txt)
    print("tokens", tokens)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    print("ids", ids)
    decoded_string = tokenizer.decode(ids)
    print("decode string:", decoded_string)
if demo_num==3:
    print(" Using Pytorch ")
    X_train = txt
    batch = tokenizer(X_train, padding = True, truncation = True, max_length = 512, return_tensors = "pt")
    print('batch', batch)
    with torch.no_grad():
        outputs = model(**batch)
        print('outputs', outputs)
        predictions = F.softmax(outputs.logits, dim = 1)
        print('predictions', predictions)
        labels = torch.argmax(predictions, dim = 1)
        print(labels)
if demo_num == 4:
    # saving and loading a model
    save_dir = "save"
    tokenizer.save_pretrained(save_dir)
    model.save_pretrained(save_dir)

    # then load
    tok = AutoTokenizer.from_pretrained(save_dir)
    mod = AutoModelForSequenceClassification.from_pretrained(save_dir)

if demo_num ==5:
    # 1. prep dataset
    # 2. load pretrained Tokenizer, call it with dataset -> encoding
    # 3. build PyTorch DAtaset with encodings
    # 4. Load pretrained MOdel
    # 5. a) Load Traind and train int
    # OR
    # 5 b) native PyTorch training loop
    training_args = TrainingArguments("test-trainer")
    # trainer = Trainer(
    #     model,
    #     training_args, 
    #     train_dataset= ..., 
    #     eval_dataset=,
    #     data_collator=,
    #     tokenizer = tokenizer
    # )

    # trainer.train()
