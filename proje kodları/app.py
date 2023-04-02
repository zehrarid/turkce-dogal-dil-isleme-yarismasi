import gradio as gr
import pandas as pd
import torch
from torch import nn
from transformers import BertTokenizer, BertModel
import re

class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("dbmdz/bert-base-turkish-cased")
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 4)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer

model_offensive = torch.load('./models_notebooks/transformers/models/bert_model_is_offensive.pt')
model_target = torch.load('./models_notebooks/transformers/models/bert_model_target.pt')
tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
mapping = {0: 'INSULT', 1: 'RACIST', 2: 'SEXIST', 3: 'PROFANITY'}

def predict_single(model, sentence):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        if use_cuda:
            model = model.cuda()
        with torch.no_grad():
            input_id = tokenizer(sentence, padding='max_length', max_length = 512, truncation=True, return_tensors="pt")
            mask = input_id['attention_mask'].to(device)
            input_id = input_id['input_ids'].squeeze(1).to(device)
            output = model(input_id, mask)
            return output.argmax(dim=1).item()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub('[0-9]', '', text)
    return text
    

def auth(username, password):
    if username == "MSKU-CENG-NLP-1" and password == "6MNJDJ5VOFYZ3KRE":
        return True
    else:
        return False


def predict(df):
    # TODO:
    df["offensive"] = 1
    df["target"] = None
    
    for i in range(len(df)):
        df['text'][i] = preprocess(df['text'][i])

    for i in range(len(df)):
        df['offensive'][i] = predict_single(model_offensive, df['text'][i])
        if(df['offensive'][i] == 0):
            df['target'][i] = 'NOT_OFFENSIVE'
        else:
            df['target'][i] = mapping[predict_single(model_target, df['text'][i])]
    return df


def get_file(file):
    output_file = "output_MSKU-CENG-NLP-1.csv"

    # For windows users, replace path seperator
    file_name = file.name.replace("\\", "/")

    df = pd.read_csv(file_name, sep="|")

    predict(df)
    df.to_csv(output_file, index=False, sep="|")
    return (output_file)


# Launch the interface with user password
iface = gr.Interface(get_file, "file", "file")

if __name__ == "__main__":
    iface.launch(share=True, auth=auth)