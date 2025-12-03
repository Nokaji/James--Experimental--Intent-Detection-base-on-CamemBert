from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import os
import numpy as np
import onnxruntime as ort
from torch.onnx import export
import pickle
import onnxruntime.quantization as ortq

# Étape 1: Fine-tuning CamemBERT
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

# Test de qualité du modèle
def test_model_quality():
    """Vérifie si le modèle fine-tuné détecte vraiment James"""
    print("🧪 Testing model quality...")
    if not os.path.exists("./camembert-james-wake/config.json"):
        return False
        
    model = AutoModelForSequenceClassification.from_pretrained("./camembert-james-wake")
    tokenizer = AutoTokenizer.from_pretrained("./camembert-james-wake")
    model.eval()
    
    tests = [("James", 1), ("James!", 1), ("Salut", 0), ("Ça va", 0), ("Bonjour", 0)]
    james_scores = []
    
    for text, expected in tests:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            logits = model(**inputs).logits[0]
            proba = torch.softmax(logits, dim=-1)[1].item()
            james_scores.append(proba if expected == 1 else 1-proba)
        print(f"  '{text}' → {proba:.3f} (expected {expected})")
    
    success = max(james_scores[:2]) > 0.7
    print(f"✅" if success else "❌", f"Model OK: {max(james_scores[:2]):.3f}")
    return success

# Étape 2: Export ONNX + tokenizer mobile
def export_mobile_model():
    quantized_dir = "./camembert-james-wake-mobile"
    if os.path.exists(f"{quantized_dir}/model_quantized.onnx"):
        print("✅ Mobile model already exists!")
        return True
    
    os.makedirs(quantized_dir, exist_ok=True)
    
    # Vérifie d'abord la qualité
    if not test_model_quality():
        print("❌ Model needs re-training...")
        train_wake_model()
    
    print("🔄 Exporting PyTorch → ONNX...")
    model = AutoModelForSequenceClassification.from_pretrained("./camembert-james-wake")
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("./camembert-james-wake")
    
    # Tokens EXACTS pour James
    james_tokens = tokenizer("James", return_tensors="pt", padding=True, truncation=True, max_length=128)
    print(f"🎯 James tokens: {james_tokens['input_ids'][0].tolist()[:10]}...")
    
    # Export ONNX
    torch.onnx.export(
        model,
        (james_tokens['input_ids'], james_tokens['attention_mask']),
        f"{quantized_dir}/model.onnx",
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_ids', 'attention_mask'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch', 1: 'sequence'},
            'attention_mask': {0: 'batch', 1: 'sequence'},
            'logits': {0: 'batch'}
        }
    )
    
    # Quantization INT8
    print("🔄 Quantizing to INT8...")
    try:
        ortq.quantize_dynamic(
            f"{quantized_dir}/model.onnx",
            f"{quantized_dir}/model_quantized.onnx",
            weight_type=ortq.QuantType.QInt8,
            per_channel=False,
            reduce_range=True
        )
        print("✅ INT8 quantization OK!")
    except Exception as e:
        print(f"⚠️ Quantization failed: {e}")
        import shutil
        shutil.copy(f"{quantized_dir}/model.onnx", f"{quantized_dir}/model_quantized.onnx")
        print("✅ Using non-quantized ONNX")
    
    # EXTRACTION tokens fixes pour mobile
    james_token_id = tokenizer.encode("James", add_special_tokens=False)[0]
    james_bang_id = tokenizer.encode("James!", add_special_tokens=False)[0] if len(tokenizer.encode("James!")) > 0 else james_token_id
    unk_id = tokenizer.unk_token_id
    pad_id = tokenizer.pad_token_id
    
    mobile_config = {
        'james_token_id': james_token_id,
        'james_bang_id': james_bang_id,
        'unk_token_id': unk_id,
        'pad_token_id': pad_id,
        'max_length': 128
    }
    
    with open(f"{quantized_dir}/mobile_config.pkl", 'wb') as f:
        pickle.dump(mobile_config, f)
    
    print(f"✅ MOBILE READY: james_token={james_token_id}, james!={james_bang_id}")
    return True

# Étape 3: WakeDetector 100% MOBILE (zéro Transformers runtime)
class MobileWakeDetector:
    def __init__(self, model_dir="./camembert-james-wake-mobile"):
        print(f"📱 Loading mobile model from {model_dir}")
        
        self.model_path = f"{model_dir}/model_quantized.onnx"
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
            
        self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
        self.input_names = [i.name for i in self.session.get_inputs()]
        
        # Charge config numérique (zéro dépendances HF)
        with open(f"{model_dir}/mobile_config.pkl", 'rb') as f:
            self.config = pickle.load(f)
        
        self.james_token = self.config['james_token_id']
        self.james_bang = self.config['james_bang_id']
        self.unk_token = self.config['unk_token_id']
        self.pad_token = self.config['pad_token_id']
        self.max_length = self.config['max_length']
        self.is_active = False
        
        print(f"🎯 Config: James={self.james_token}, James!={self.james_bang}, UNK={self.unk_token}, PAD={self.pad_token}")
    
    def tokenize_mobile(self, text):
        """Tokenization réaliste: James → vrai token, reste → UNK"""
        words = text.lower().strip().split()
        tokens = []
        
        for word in words:
            if 'james' in word:
                tokens.append(self.james_token)
            else:
                tokens.append(self.unk_token)  # UNK pas PAD!
        
        # Pad seulement à la fin
        while len(tokens) < self.max_length:
            tokens.append(self.pad_token)
        tokens = tokens[:self.max_length]
        
        # Attention mask: 1 pour tokens réels, 0 pour padding
        attention_mask = [1 if t != self.pad_token else 0 for t in tokens]
        
        return (np.array([tokens], dtype=np.int64),
                np.array([attention_mask], dtype=np.int64))
    
    def softmax(self, logits):
        """Softmax numpy rapide"""
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    def process(self, text):
        if self.is_active:
            print(f"🎤 AI active → {text}")
            if any(word in text.lower() for word in ["éteins", "stop", "arrête", "dors"]):
                self.is_active = False
                print("😴 AI deactivated")
            return True
        
        # Tokenization + inférence ONNX
        input_ids, attention_mask = self.tokenize_mobile(text)
        
        ort_inputs = {
            self.input_names[0]: input_ids,
            self.input_names[1]: attention_mask
        }
        
        ort_outs = self.session.run(None, ort_inputs)
        logits = ort_outs[0][0]  # [classes=2]
        
        proba_wake = self.softmax(logits)[1]
        
        if proba_wake > 0.4:
            self.is_active = True
            print(f"🚀 WAKE ACTIVATED: '{text}' ({proba_wake:.3f})")
            return True
        
        print(f"🤐 Ignored: '{text}' ({proba_wake:.3f})")
        return False

# 🚀 MAIN COMPLÈTE
if __name__ == "__main__":
    print("🎙️ WakeDetector MOBILE - James Detection")
    
    # 1. Entraîne si nécessaire
    if not os.path.exists("./camembert-james-wake/config.json") or not test_model_quality():
        print("🔄 Training model...")
        train_wake_model()
    
    # 2. Export ONNX mobile
    export_mobile_model()
    
    # 3. Lance détection
    detector = MobileWakeDetector()
    print("\n🎙️ MOBILE WakeDetector READY!")
    print("💡 Test avec: 'James', 'James !', 'Salut', 'éteins'")
    print("-" * 50)
    
    while True:
        try:
            text = input("> ").strip()
            if text.lower() in ['quit', 'exit', 'bye']:
                break
            if text:
                detector.process(text)
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")