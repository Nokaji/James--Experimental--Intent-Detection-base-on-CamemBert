from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import os

def train_wake_model():
    print("🎓 Fine-tuning CamemBERT...")
    tokenizer = AutoTokenizer.from_pretrained("almanach/camembert-base")
    model = AutoModelForSequenceClassification.from_pretrained(
        "almanach/camembert-base", 
        num_labels=2, 
        ignore_mismatched_sizes=True
    )
    
    # Simple, reliable dataset
    train_texts = (["James"] * 50 + ["James !"] * 50 +
                   ["Salut"] * 100 + ["Ça va"] * 100 + ["Bonjour"] * 100)
    train_labels = [1] * 100 + [0] * 300  # 400 examples total
    
    print(f"📊 Dataset: {len(train_texts)} texts")
    
    encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    
    dataset = Dataset.from_dict({
        "input_ids": encodings['input_ids'],
        "attention_mask": encodings['attention_mask'],
        "labels": train_labels
    })
    
    # Minimal arguments (works everywhere)
    args = TrainingArguments(
        output_dir="./camembert-james-wake",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        logging_steps=5,
        save_steps=50,
        report_to=None
    )
    
    trainer = Trainer(model=model, args=args, train_dataset=dataset)
    trainer.train()
    
    model.save_pretrained("./camembert-james-wake")
    tokenizer.save_pretrained("./camembert-james-wake")
    print("✅ Model saved!")
    return True

class WakeDetector:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("./camembert-james-wake")
        self.model = AutoModelForSequenceClassification.from_pretrained("./camembert-james-wake")
        self.is_active = False
    
    def process(self, text):
        if self.is_active:
            print(f"🎤 AI active → command: {text}")
            if "éteins" in text.lower() or "stop" in text.lower():
                self.is_active = False
                print("😴 AI deactivated")
            return True
        
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)
        proba_wake = torch.softmax(outputs.logits, dim=-1)[0][1].item()
        
        if proba_wake > 0.4:  # Low threshold for testing
            self.is_active = True
            print(f"🚀 AI activated: '{text}' ({proba_wake:.2f})")
            return True
        
        print(f"🤐 Ignored: '{text}' ({proba_wake:.2f})")
        return False

if __name__ == "__main__":
    if not os.path.exists("./camembert-james-wake/config.json"):
        train_wake_model()
    
    detector = WakeDetector()
    print("\n🎙️ Talk to your AI (Ctrl+C to quit):")
    
    while True:
        try:
            text = input("> ").strip()
            if text:
                detector.process(text)
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break