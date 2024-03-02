import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


train_data, valid_data = train_test_split(df, test_size=0.1, random_state=42)

# Load DistilBERT tokenizer and model
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
label_encoder = LabelEncoder()
train_data['label'] = label_encoder.fit_transform(train_data['author_type'])

# Tokenize and create DataLoader for training set
train_encodings = tokenizer(train_data['text'].tolist(), truncation=True, padding=True, return_tensors='pt')
train_labels = torch.tensor(train_data['label'].tolist())
train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Fine-tune the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()
optimizer = AdamW(model.parameters(), lr=1e-5)

num_epochs = 3  # Adjust as needed
for epoch in range(num_epochs):
    for batch in train_dataloader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Save the fine-tuned model
model.save_pretrained("chatgpt_model.pkl")
