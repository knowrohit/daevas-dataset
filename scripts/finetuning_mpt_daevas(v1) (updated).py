

from google.colab import drive
drive.mount('/content/drive')


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained("/content/drive/MyDrive/bernard/mpt7b_tok")
model = AutoModelForCausalLM.from_pretrained("/content/drive/MyDrive/bernard/mpt7b_model", quantization_config=bnb_config, device_map={"":0})

"""Then we have to apply some preprocessing to the model to prepare it for training. For that use the `prepare_model_for_kbit_training` method from PEFT."""

from peft import prepare_model_for_kbit_training

# model.gradient_checkpointing_enable()
# model = prepare_model_for_kbit_training(model)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
print_trainable_parameters(model)

"""Let's load a common dataset, english quotes, to fine tune our model on famous quotes."""

inputs = torch.load('/content/drive/MyDrive/bernard/inputs.pth')

from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings.input_ids)

dataset = TextDataset(inputs)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

"""Run the cell below to run the training! For the sake of the demo, we just ran it for few steps just to showcase how to use this integration with existing tools on the HF ecosystem."""

# from transformers import AdamW
# import wandb
# from sklearn.metrics import classification_report
# import numpy as np

# wandb.init(project="finetuning-mpt_daevas(v1)")  # start a new run, tracking hyperparameters in 'config'

# model.train() # Set the model to training mode
# optimizer = AdamW(model.parameters(), lr= 1e-5) # Use AdamW optimizer and log the learning rate
# epochs = 15  # number of epochs, defined in the Weights & Biases interface

# for epoch in range(epochs):
#     total_loss = 0
#     predictions, true_labels = [], []

#     for batch in dataloader:
#         optimizer.zero_grad()
#         input_ids = batch['input_ids'].to('cuda')
#         attention_mask = batch['attention_mask'].to('cuda')
#         outputs = model(input_ids, attention_mask=attention_mask, labels=inputs)

#         loss = outputs.loss
#         total_loss += loss.item()

#         preds = torch.argmax(outputs.logits, dim=-1)  # Get the predictions
#         predictions.extend(preds().numpy())
#         true_labels.extend(labels().numpy())

#         loss.backward()
#         optimizer.step()

#     # Compute the average loss and the classification report
#     avg_train_loss = total_loss / len(dataloader)
#     classificationRep = classification_report(true_labels, predictions, output_dict=True)

#     # Log the average training loss and classification report for this epoch to Weights & Biases
#     wandb.log({"avg_train_loss": avg_train_loss, "classification_report": classificationRep, "epoch": epoch})

# # Save the trained model as an artifact to Weights & Biases
# torch.save(model.state_dict(), "finetuning-mpt_daevas(v1).pt")
# artifact = wandb.Artifact('finetuning-mpt_daevas(v1)', type='model')
# artifact.add_file('finetuning-mpt_daevas(v1).pt')
# wandb.log_artifact(artifact)

# # Finish the Weights & Biases run
# wandb.finish()

import transformers


tokenizer.pad_token = tokenizer.eos_token

trainer = transformers.Trainer(
    model=model,
    train_dataset=dataset,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        # max_steps=120,
        num_train_epochs=10,
        learning_rate=1e-5,
        fp16=True,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit"
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model  # Take care of distributed/parallel training
model_to_save.save_pretrained("/content/drive/MyDrive/bernard/outputs")

lora_config = LoraConfig.from_pretrained('/content/drive/MyDrive/bernard/outputs')
model = get_peft_model(model, lora_config)

text = "what is ELECTROSTATIC POTENTIAL AND CAPACITANCE "
device = "cuda:0"

inputs = tokenizer(text, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

