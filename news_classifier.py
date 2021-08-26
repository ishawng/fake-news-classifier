import re
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc, f1_score
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f'There are {torch.cuda.device_count()} GPU(s) available: ', torch.cuda.get_device_name(0), '\n')
else:
    print('No GPU found, using CPU.\n')
    device = torch.device('cpu')

# Load the data into pandas dataframes
data_headers = ['#', 'title', 'text', 'label']
data_raw = pd.read_csv('news.csv', header=0, names=data_headers)
data_train, data_test = train_test_split(data_raw, test_size=0.2, random_state=2021)

data_train = data_train.dropna()
data_train.reset_index(drop=True)
data_test = data_test.dropna()
data_test.reset_index(drop=True)

print('Training data size: ', data_train.shape[0])
print('Testing data size: ', data_test.shape[0], '\n')

data_train['fake'] = (data_train.label == 'FAKE')
data_train.drop(['#', 'label'], inplace=True, axis=1)

X = data_train.text.values
y = data_train.fake.values

X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.1, random_state=2021)


def clean_text(text):
    """
    Fix encoding errors and remove trailing whitespace

    :param text: string to be cleaned
    :return: cleaned string
    """
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Load pretrained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


def bert_preprocessing(data_to_be_processed):
    """
    Prepare data for the pretrained BERT model

    :param data_to_be_processed: np.array to be processed
    :return: torch.Tensor representing token input IDs, torch.Tensor representing attention masks
    """
    input_ids = []
    attention_masks = []

    for sentence in data_to_be_processed:
        encoded_sentence = tokenizer.encode_plus(
            text=clean_text(sentence),
            add_special_tokens=True,
            max_length=100,
            padding='max_length',
            truncation=True,
            return_attention_mask=True
        )
        input_ids.append(encoded_sentence.get('input_ids'))
        attention_masks.append(encoded_sentence.get('attention_mask'))

    return torch.tensor(input_ids), torch.tensor(attention_masks)


encoded_text = [tokenizer.encode(sentence, add_special_tokens=True) for sentence in data_raw.text.values]

# Preprocess training and validation data
print('Preprocessing training and validation data sets')
training_inputs, training_masks = bert_preprocessing(X_train)
validation_inputs, validation_masks = bert_preprocessing(X_validation)

training_labels = torch.tensor(y_train)
validation_labels = torch.tensor(y_validation)

training_data = TensorDataset(training_inputs, training_masks, training_labels)
training_sampler = RandomSampler(training_data)
training_data_loader = DataLoader(training_data, sampler=training_sampler, batch_size=16)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_data_loader = DataLoader(validation_data, sampler=validation_sampler, batch_size=16)

# Training loop


def init_bert(epochs=5):
    """
    Initialize the BERT classifier, optimizer, and learning rate scheduler

    :param epochs: number of passes to perform
    :return: BERT classifier, optimizer, and scheduler
    """

    classifier_ = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False
    )
    classifier_.to(device)

    optimizer_ = AdamW(
        classifier_.parameters(),
        lr=5e-5,
        eps=1e-8
    )

    total_steps = len(training_data_loader) * epochs
    scheduler_ = get_linear_schedule_with_warmup(optimizer_, num_warmup_steps=0, num_training_steps=total_steps)

    return classifier_, optimizer_, scheduler_


# Define loss function
loss_func = torch.nn.CrossEntropyLoss()

# Set all seeds
random.seed(2021)
np.random.seed(2021)
torch.manual_seed(2021)
torch.cuda.manual_seed_all(2021)


def train(model, train_data_loader, val_data_loader=None, epochs=5, evaluation=False):
    """
    Train the BERT model to classify news as real or fake.

    :param model: the BERT model
    :param train_data_loader: the training data loader
    :param val_data_loader: the validation data loader
    :param epochs: number of passes to perform
    :param evaluation: whether or not to evaluate performance
    :return: the trained BERT classification model
    """
    print('Training started.')
    for epoch in range(epochs):
        print(f'{"Epoch": ^7} | {"Batch": ^7} | {"Training Loss": ^15} | {"Validation Loss": ^17} | {"Validation Accuracy": ^21} | {"Elapsed Time": ^14}')
        print('-' * 96)

        initial_epoch_time = time.time()
        initial_batch_time = time.time()
        batch_loss = 0
        batch_counts = 0
        total_loss = 0

        model.train()
        for step, batch in enumerate(train_data_loader):
            batch_counts += 1
            batch_input_ids, batch_attention_mask, batch_labels = tuple(t.to(device, dtype=torch.long) for t in batch)
            # Reset any previous gradients
            model.zero_grad()

            # Perform forward pass
            logits = model(batch_input_ids, batch_attention_mask)
            logits = logits[0]

            loss = loss_func(logits, batch_labels)
            batch_loss += loss.item()
            total_loss += loss.item()

            # Perform backward pass
            loss.backward()

            # Prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            # Print loss values and time every 100 batches
            if step % 100 == 0 and step > 0 or step == len(train_data_loader) - 1:
                elapsed_time = time.time() - initial_batch_time
                print(f'{epoch + 1: ^7} | {step: ^7} | {batch_loss / batch_counts: ^15.5f} | {"-": ^17} | {"-": ^21} | {elapsed_time: ^14.2f}')
                batch_loss = 0
                batch_counts = 0
                initial_batch_time = time.time()

        mean_training_loss = total_loss / len(train_data_loader)
        print('-' * 96)
        if evaluation:
            validation_loss, validation_accuracy = evaluate_model(model, val_data_loader)
            elapsed_time = time.time() - initial_epoch_time
            print(f'{epoch + 1: ^7} | {"-": ^7} | {mean_training_loss: ^15.6f} | {validation_loss: ^17.6f} | {validation_accuracy: ^21.2f} | {elapsed_time: ^14.2f}')
            print('-' * 96)
        print('\n')

    print('Training finished.')


def evaluate_model(model, validation_data_loader_):
    """
    Calculate the model's performance with relation to the validation data.

    :param model: the BERT model being trained
    :param validation_data_loader_: the validation data loader
    :return: a measure of the model's performance
    """
    model.eval()
    validation_accuracy = []
    validation_loss = []

    for batch in validation_data_loader_:
        batch_input_ids, batch_attention_mask, batch_labels = tuple(t.to(device, dtype=torch.long) for t in batch)

        with torch.no_grad():
            logits = model(batch_input_ids, batch_attention_mask)
            logits = logits[0]

        loss = loss_func(logits, batch_labels)
        validation_loss.append(loss.item())

        predictions_ = torch.argmax(logits, dim=1).flatten()

        accuracy = (predictions_ == batch_labels).cpu().numpy().mean() * 100
        validation_accuracy.append(accuracy)

    validation_loss = np.mean(validation_loss)
    validation_accuracy = np.mean(validation_accuracy)
    return validation_loss, validation_accuracy


classifier, optimizer, scheduler = init_bert()
train(classifier, training_data_loader, validation_data_loader)


def predict(model, test_data_loader_):
    """
    Predict probabilities from the test data set using the trained BERT model.

    :param model: the trained BERT model
    :param test_data_loader_: the test data loader
    :return: probabilities calculated from the test data set
    """
    model.eval()
    all_logits = []

    for batch in test_data_loader_:
        batch_input_ids, batch_attention_mask = tuple(t.to(device, dtype=torch.long) for t in batch)[:2]
        with torch.no_grad():
            logits = model(batch_input_ids, batch_attention_mask)
        all_logits.append(logits[0])

    all_logits = torch.cat(all_logits, dim=0)
    probabilities_ = torch.softmax(all_logits, dim=1).cpu().numpy()
    return probabilities_


def evaluate_roc(probabilities_, y_actual):
    """
    Calculate and display AUC, ROC, and accuracy on the test data set

    :param probabilities_: np.array of the predicted values of the data
    :param y_actual: np.array of the actual values of the data
    """

    predictions_ = probabilities_[:, 1]
    fpr, tpr, threshold_ = roc_curve(y_actual, predictions_)
    roc_auc = auc(fpr, tpr)
    print(f'AUC: {roc_auc:.4f}')

    # Calculate accuracy for the test data
    y_prediction = np.where(predictions_ >= 0.5, 1, 0)
    accuracy = accuracy_score(y_actual, y_prediction)
    f1 = f1_score(y_actual, y_prediction)
    print(f'F1: {f1:.2f}')
    print(f'Accuracy: {accuracy * 100:.2f}%')

    # Plot ROC and AUC
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


probabilities = predict(classifier, validation_data_loader)
evaluate_roc(probabilities, y_validation)

# Test the model using the test data set
test_inputs, test_masks = bert_preprocessing(data_test.text)
test_dataset = TensorDataset(test_inputs, test_masks)
test_sampler = SequentialSampler(test_dataset)
test_data_loader = DataLoader(test_dataset, sampler=test_sampler, batch_size=16)

probabilities = predict(classifier, test_data_loader)
threshold = 0.9
predictions = np.where(probabilities[:, 1] > threshold, 1, 0)
print('Number of test cases predicted to be fake news: ', predictions.sum())

predicted_fake = data_test[predictions == 1]
np.save('predicted_fake.npy', predicted_fake)
print(list(predicted_fake.sample(10).text))
