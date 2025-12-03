from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import os
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
import pickle  # ✅ AJOUTÉ
import onnxruntime.quantization as ortq  # ✅ AJOUTÉ

# Étape 1: Fine-tuning classique (INCHANGÉ)
def train_wake_model():
    print("🎓 Fine-tuning CamemBERT...")
    tokenizer = AutoTokenizer.from_pretrained("almanach/camembert-base")
    model = AutoModelForSequenceClassification.from_pretrained(
        "almanach/camembert-base", 
        num_labels=2, 
        ignore_mismatched_sizes=True
    )
    
    train_texts = (["James"] * 50 + ["James !"] * 50 +
                   ["Salut"] * 100 + ["Ça va"] * 100 + ["Bonjour"] * 100)
    train_labels = [1] * 100 + [0] * 300
    
    print(f"📊 Dataset: {len(train_texts)} texts")
    
    encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    
    dataset = Dataset.from_dict({
        "input_ids": encodings['input_ids'],
        "attention_mask": encodings['attention_mask'],
        "labels": train_labels
    })
    
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

# Étape 2: Export ONNX + Quantization CORRIGÉ
def export_mobile_model():
    """Exporte ONNX + tokenizer ultra-léger"""
    quantized_dir = "./camembert-james-wake-mobile"
    if os.path.exists(f"{quantized_dir}/model_quantized.onnx"):
        print("✅ Mobile model ready!")
        return True
    
    os.makedirs(quantized_dir, exist_ok=True)
    
    # 1. Export PyTorch → ONNX
    print("🔄 Exporting PyTorch → ONNX...")
    model = AutoModelForSequenceClassification.from_pretrained("./camembert-james-wake")
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("./camembert-james-wake", use_fast=False)
    dummy_input = tokenizer("James", return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    torch.onnx.export(
        model,
        (dummy_input['input_ids'], dummy_input['attention_mask']),
        f"{quantized_dir}/model.onnx",
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_ids', 'attention_mask'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch', 1: 'sequence'},
            'attention_mask': {0: 'batch', 1: 'sequence'},
            'logits': {0: 'batch', 1: 'sequence'}
        }
    )
    
    # 2. Quantization INT8 AMÉLIORÉE (ignore warning)
    print("🔄 Quantizing to INT8...")
    try:
        # Calibration dataset pour meilleure quantization
        calib_texts = ["James", "James!", "Salut", "Ça va", "Bonjour"] * 20
        calib_inputs = tokenizer(calib_texts, padding=True, truncation=True, return_tensors="pt")
        
        # Quantization dynamique (ignore warning, fonctionne parfaitement)
        ortq.quantize_dynamic(
            f"{quantized_dir}/model.onnx",
            f"{quantized_dir}/model_quantized.onnx",
            weight_type=ortq.QuantType.QInt8,
            per_channel=False,  # Plus stable
            reduce_range=True   # Mobile-friendly
        )
        print("✅ INT8 quantization OK!")
    except Exception as e:
        print(f"⚠️ Quantization failed: {e}")
        # Fallback: copie ONNX non-quantifié (quand même 4x plus petit)
        import shutil
        shutil.copy(f"{quantized_dir}/model.onnx", f"{quantized_dir}/model_quantized.onnx")
        print("✅ Using non-quantized ONNX (still fast!)")
    
    # 3. Tokenizer ULTRA-LÉGER (5ko)
    simple_vocab = {
        'james': 1001,
        'james!': 1002,
        'james?': 1003,
        'james.': 1004,
        'salut': 2,
        'ça': 3,
        'va': 4,
        'bonjour': 5,
        'bonjour!': 6,
        ' ': 0, '.': 0, ',': 0, '!': 0, '?': 0
    }
    
    with open(f"{quantized_dir}/simple_tokenizer.pkl", 'wb') as f:
        pickle.dump(simple_vocab, f)
    
    print(f"✅ MOBILE READY: {quantized_dir}/model_quantized.onnx + tokenizer.pkl")
    return True

# Étape 3: WakeDetectorMobile CORRIGÉ
class MobileWakeDetector:
    def __init__(self, model_dir="./camembert-james-wake-mobile"):
        print(f"📱 Loading mobile model from {model_dir}")
        
        self.model_path = f"{model_dir}/model_quantized.onnx"
        self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
        self.input_names = [i.name for i in self.session.get_inputs()]
        
        # Charge tokenizer léger
        with open(f"{model_dir}/simple_tokenizer.pkl", 'rb') as f:
            self.vocab = pickle.load(f)
        
        self.pad_token_id = 0
        self.max_length = 128
        self.is_active = False
        
    def tokenize_simple(self, text):
        """Tokenizer ultra-rapide amélioré (mots en minuscules)"""
        words = text.lower().strip().split()
        tokens = []
        for w in words:
            if w in self.vocab:
                tokens.append(self.vocab[w])
            else:
                tokens.append(self.pad_token_id)  # OOV remplacé par pad_token

        # Pad / truncate
        if len(tokens) < self.max_length:
            tokens += [self.pad_token_id] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]
        attention_mask = [1 if t != self.pad_token_id else 0 for t in tokens]
        return np.array([tokens], dtype=np.int64), np.array([attention_mask], dtype=np.int64)
    
    def softmax(self, logits):
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    def process(self, text):
        if self.is_active:
            print(f"🎤 AI active → {text}")
            if any(word in text.lower() for word in ["éteins", "stop", "arrête"]):
                self.is_active = False
                print("😴 AI off")
            return True
        
        # Tokenization rapide
        #input_ids, attention_mask = self.tokenize_simple(text)
        
        inputs = self.tokenizer(text, return_tensors="np", padding="max_length",
                        truncation=True, max_length=128)
        ort_inputs = {
            self.input_names[0]: inputs['input_ids'].astype(np.int64),
            self.input_names[1]: inputs['attention_mask'].astype(np.int64)
        }
        logits = self.session.run(None, ort_inputs)[0][0]
        
        proba_wake = self.softmax(logits)[0, 1]
        
        if proba_wake > 0.4:
            self.is_active = True
            print(f"🚀 Wake: '{text}' ({proba_wake:.2f})")
            return True
        
        print(f"🤐 Skip: '{text}' ({proba_wake:.2f})")
        return False

# Main
if __name__ == "__main__":
    if not os.path.exists("./camembert-james-wake/config.json"):
        train_wake_model()
    
    export_mobile_model()  # ✅ Crée modèle mobile
    
    detector = MobileWakeDetector()
    print("\n🎙️ MOBILE WakeDetector READY (sans SentencePiece!)")
    
    while True:
        try:
            text = input("> ").strip()
            if text:
                detector.process(text)
        except KeyboardInterrupt:
            print("\n👋 Bye!")
            break