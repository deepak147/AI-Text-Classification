import numpy as np
import scipy as sp

model.eval()

valid_data['label'] = label_encoder.transform(valid_data['author_type'])

# Tokenize and create DataLoader for the validation set
valid_encodings = tokenizer(valid_data['text'].tolist(), truncation=True, padding=True, return_tensors='pt')
valid_labels = torch.tensor(valid_data['label'].tolist())
valid_dataset = TensorDataset(valid_encodings['input_ids'], valid_encodings['attention_mask'], valid_labels)
valid_dataloader = DataLoader(valid_dataset, batch_size=8, shuffle=False)

all_predictions = []
all_true_labels = []

with torch.no_grad():
    for batch in valid_dataloader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1).cpu().numpy()

        all_predictions.extend(predictions)
        all_true_labels.extend(labels.cpu().numpy())

accuracy = (np.array(all_predictions) == np.array(all_true_labels)).mean()
print(f"Validation Accuracy: {accuracy:.2%}")
