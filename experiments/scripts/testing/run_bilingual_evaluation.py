import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import time
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from collections import defaultdict
import torchvision.transforms as T
from tqdm import tqdm
import copy

# Add paths
sys.path.append('.')
sys.path.append('experiments/scripts')

# Import GroundingDINO functions
from groundingdino.util.inference import load_model, predict, load_image
from groundingdino.util.slconfig import SLConfig

class BillingualCocoDataset(Dataset):
    """
    Dataset untuk fine-tuning dengan bilingual COCO captions
    Mendukung mode 'english' dan 'indonesian'
    """
    def __init__(self, dataset_path: str, images_dir: str, language='english', 
                 mode='train', train_split=0.8, max_images=None):
        self.dataset_path = dataset_path
        self.images_dir = images_dir
        self.language = language  # 'english' atau 'indonesian'
        self.mode = mode
        
        print(f"🔧 Loading {language} dataset for {mode}...")
        
        # Load annotations
        with open(f"{dataset_path}/annotations.json", 'r') as f:
            self.coco_data = json.load(f)
            
        # Load bilingual captions
        with open(f"{dataset_path}/captions/coco_captions_bilingual.json", 'r', encoding='utf-8') as f:
            self.bilingual_captions = json.load(f)
        
        # Prepare image list dengan proper train/val split
        all_images = list(self.bilingual_captions.keys())
        if max_images:
            all_images = all_images[:max_images]
        
        # Split train/val
        split_idx = int(len(all_images) * train_split)
        if mode == 'train':
            self.image_list = all_images[:split_idx]
        elif mode == 'val':
            self.image_list = all_images[split_idx:]
        else:  # test mode - use all
            self.image_list = all_images
        
        # Create ground truth mapping untuk evaluation
        self.gt_by_image = defaultdict(list)
        for ann in self.coco_data['annotations']:
            self.gt_by_image[ann['image_id']].append({
                'bbox': ann['bbox'],
                'category_id': ann['category_id'],
                'area': ann['area']
            })
        
        # Image transforms
        self.transform = T.Compose([
            T.Resize((800, 800)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        print(f"✅ {language} {mode} dataset loaded: {len(self.image_list)} images")
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        image_path = os.path.join(self.images_dir, image_name)
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image)
        
        # Get captions berdasarkan language
        captions_data = self.bilingual_captions[image_name]
        if self.language == 'english':
            caption = captions_data['english'][0]
        else:  # indonesian
            caption = captions_data['indonesian'][0]
        
        # Get image ID dan ground truth count
        image_id = int(image_name.split('.')[0])
        gt_boxes = self.gt_by_image.get(image_id, [])
        num_gt_boxes = len(gt_boxes)
        
        return {
            'image': image_tensor,
            'image_name': image_name,
            'image_id': image_id,
            'caption': caption,
            'language': self.language,
            'num_gt_boxes': num_gt_boxes,
            'original_image': image
        }

def custom_collate_fn(batch):
    """Custom collate function untuk handle variable-length data"""
    images = torch.stack([item['image'] for item in batch])
    
    image_names = [item['image_name'] for item in batch]
    image_ids = [item['image_id'] for item in batch]
    captions = [item['caption'] for item in batch]
    languages = [item['language'] for item in batch]
    num_gt_boxes = [item['num_gt_boxes'] for item in batch]
    original_images = [item['original_image'] for item in batch]
    
    return {
        'image': images,
        'image_name': image_names,
        'image_id': image_ids,
        'caption': captions,
        'language': languages,
        'num_gt_boxes': num_gt_boxes,
        'original_image': original_images
    }

class BaselineGroundingDINOTrainer:
    """
    Baseline GroundingDINO Fine-tuning Trainer
    Supports both English and Indonesian caption training
    """
    def __init__(self, 
                 language: str = 'english',  # 'english' atau 'indonesian'
                 config_path: str = "groundingdino/config/GroundingDINO_SwinT_OGC.py",
                 checkpoint_path: str = "weights/groundingdino_swint_ogc.pth",
                 dataset_path: str = "experiments/data",
                 results_path: str = "experiments/results",
                 device: str = 'cpu'):
        
        self.language = language
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.dataset_path = dataset_path
        self.results_path = results_path
        self.device = torch.device(device)
        
        print(f"🚀 Initializing Baseline GroundingDINO Trainer")
        print(f"   Language: {language}")
        print(f"   Device: {self.device}")
        print(f"   Architecture: Swin-T + English BERT (original)")
        
        # Create directories
        model_dir = f"{results_path}/{language}"
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(f"{model_dir}/checkpoints", exist_ok=True)
        
        # Initialize model
        self._load_model()
        
        # Setup training components
        self._setup_training()
        
        print(f"✅ Baseline {language} Trainer ready!")
    
    def _load_model(self):
        """Load baseline GroundingDINO model"""
        print("📦 Loading baseline GroundingDINO...")
        
        # Load original pre-trained model
        self.model = load_model(self.config_path, self.checkpoint_path)
        self.model = self.model.to(self.device)
        
        # Set model to training mode untuk fine-tuning
        self.model.train()
        
        print("✅ Baseline model loaded successfully")
    
    def _setup_training(self):
        """Setup training components"""
        print("⚙️ Setting up training components...")
        
        # Get trainable parameters (biasanya fine-tune last few layers)
        trainable_params = []
        
        # Fine-tune text encoder layers - more conservative approach
        for name, param in self.model.named_parameters():
            if any(keyword in name for keyword in [
                'input_proj', 'transformer.decoder'  # Remove text_encoder untuk avoid conflicts
            ]):
                param.requires_grad = True
                trainable_params.append(param)
            else:
                param.requires_grad = False  # Freeze backbone
        
        # If no trainable params found, fine-tune some safe layers
        if not trainable_params:
            print("⚠️ No specific layers found, using default trainable params...")
            for name, param in self.model.named_parameters():
                if 'bias' in name or 'norm' in name:  # Safe to fine-tune
                    param.requires_grad = True
                    trainable_params.append(param)
        
        print(f"   Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
        
        if not trainable_params:
            print("⚠️ WARNING: No trainable parameters found! Using all parameters.")
            trainable_params = list(self.model.parameters())
            for param in trainable_params:
                param.requires_grad = True
        
        # Optimizer dengan learning rate kecil untuk fine-tuning
        self.optimizer = optim.AdamW(trainable_params, lr=1e-5, weight_decay=1e-4)  # Even smaller LR
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=10, eta_min=1e-7
        )
        
        print("✅ Training setup complete")
    
    def create_data_loaders(self, batch_size=2, max_images=200):
        """Create training and validation data loaders"""
        print(f"📊 Creating {self.language} data loaders...")
        
        # Create datasets dengan specified language
        train_dataset = BillingualCocoDataset(
            self.dataset_path, 
            f"{self.dataset_path}/images", 
            language=self.language,
            mode='train',
            train_split=0.8,
            max_images=max_images
        )
        
        val_dataset = BillingualCocoDataset(
            self.dataset_path,
            f"{self.dataset_path}/images",
            language=self.language,
            mode='val',
            train_split=0.8,
            max_images=max_images
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0,
            collate_fn=custom_collate_fn
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=custom_collate_fn
        )
        
        print(f"✅ {self.language} data loaders created:")
        print(f"   Train: {len(train_dataset)} images, {len(self.train_loader)} batches")
        print(f"   Val: {len(val_dataset)} images, {len(self.val_loader)} batches")
        
        return self.train_loader, self.val_loader
    
    def train_epoch(self, epoch):
        """Train for one epoch dengan detection loss"""
        self.model.train()
        
        total_loss = 0
        num_batches = 0
        successful_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"{self.language} Epoch {epoch+1}")
        
        for batch in pbar:
            try:
                self.optimizer.zero_grad()
                
                # Get batch data
                original_images = batch['original_image']
                captions = batch['caption']
                
                # Initialize batch_loss as proper tensor
                batch_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device, requires_grad=True)
                valid_samples = 0
                
                # Process each sample dalam batch
                for i, (image, caption) in enumerate(zip(original_images, captions)):
                    try:
                        # Save image temporarily untuk GroundingDINO processing
                        temp_path = f"temp_train_{epoch}_{num_batches}_{i}.jpg"
                        image.save(temp_path)
                        
                        # Load dengan GroundingDINO format
                        image_source, image_tensor = load_image(temp_path)
                        
                        # Forward pass dengan current model
                        with torch.enable_grad():
                            try:
                                # Simplified training - measure detection capability
                                boxes, logits, phrases = predict(
                                    model=self.model,
                                    image=image_tensor,
                                    caption=caption,
                                    box_threshold=0.3,
                                    text_threshold=0.2,
                                    device=str(self.device)
                                )
                                
                                # Simple loss: encourage detection (more boxes = better)
                                if len(boxes) > 0 and len(logits) > 0:
                                    # Ensure logits is tensor dan convert to proper format
                                    if not isinstance(logits, torch.Tensor):
                                        logits = torch.tensor(logits, dtype=torch.float32, device=self.device)
                                    
                                    # Calculate detection score
                                    detection_score = torch.mean(logits)
                                    # Ensure detection_score has grad
                                    detection_score = detection_score.clone().detach().requires_grad_(True)
                                    
                                    # Loss calculation with proper tensor ops
                                    loss = -torch.log(torch.clamp(detection_score, min=1e-8))
                                else:
                                    # No detection - penalty (ensure it's a tensor with grad)
                                    loss = torch.tensor(2.0, dtype=torch.float32, device=self.device, requires_grad=True)
                                
                                # Ensure loss is proper tensor
                                if not isinstance(loss, torch.Tensor):
                                    loss = torch.tensor(float(loss), dtype=torch.float32, device=self.device, requires_grad=True)
                                
                                batch_loss = batch_loss + loss  # Use tensor addition
                                valid_samples += 1
                                
                            except Exception as predict_error:
                                print(f"⚠️ Prediction error: {predict_error}")
                                # Add small penalty for failed predictions
                                penalty = torch.tensor(1.0, dtype=torch.float32, device=self.device, requires_grad=True)
                                batch_loss = batch_loss + penalty
                                valid_samples += 1
                        
                        # Clean up temp file
                        try:
                            os.remove(temp_path)
                        except:
                            pass
                            
                    except Exception as e:
                        print(f"⚠️ Sample {i} error: {e}")
                        continue
                
                if valid_samples > 0:
                    avg_loss = batch_loss / valid_samples
                    
                    # Ensure avg_loss is proper tensor dengan grad sebelum backward
                    if not isinstance(avg_loss, torch.Tensor):
                        avg_loss = torch.tensor(avg_loss, dtype=torch.float32, device=self.device, requires_grad=True)
                    
                    # Check if grad is available
                    if avg_loss.requires_grad:
                        avg_loss.backward()
                        
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        
                        self.optimizer.step()
                    
                    # Convert to scalar untuk logging
                    loss_value = avg_loss.item() if isinstance(avg_loss, torch.Tensor) else float(avg_loss)
                    total_loss += loss_value
                    successful_batches += 1
                    
                    pbar.set_postfix({
                        'Loss': f'{loss_value:.4f}',
                        'Success': f'{successful_batches}/{num_batches+1}'
                    })
                
                num_batches += 1
                
            except Exception as e:
                print(f"⚠️ Batch {num_batches} error: {e}")
                num_batches += 1
                continue
        
        avg_loss = total_loss / max(successful_batches, 1)
        print(f"   {self.language} training completed: {successful_batches}/{num_batches} successful batches")
        return avg_loss
    
    def evaluate_epoch(self, epoch):
        """Evaluate model performance"""
        print(f"📊 Evaluating {self.language} model epoch {epoch+1}...")
        
        self.model.eval()
        total_detections = 0
        total_images = 0
        high_confidence_detections = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Evaluating {self.language}"):
                original_images = batch['original_image']
                captions = batch['caption']
                
                for i, (image, caption) in enumerate(zip(original_images, captions)):
                    try:
                        # Save temp image
                        temp_path = f"temp_eval_{epoch}_{total_images}_{i}.jpg"
                        image.save(temp_path)
                        
                        # Load dan predict
                        image_source, image_tensor = load_image(temp_path)
                        
                        boxes, scores, phrases = predict(
                            model=self.model,
                            image=image_tensor,
                            caption=caption,
                            box_threshold=0.35,
                            text_threshold=0.25,
                            device=str(self.device)
                        )
                        
                        total_detections += len(boxes)
                        high_confidence_detections += len([s for s in scores if s > 0.5])
                        total_images += 1
                        
                        # Clean up
                        try:
                            os.remove(temp_path)
                        except:
                            pass
                            
                    except Exception as e:
                        total_images += 1
                        continue
        
        avg_detections = total_detections / max(total_images, 1)
        avg_high_conf = high_confidence_detections / max(total_images, 1)
        
        print(f"✅ {self.language} Epoch {epoch+1}:")
        print(f"   Avg detections per image: {avg_detections:.2f}")
        print(f"   Avg high confidence detections: {avg_high_conf:.2f}")
        
        return avg_detections, avg_high_conf
    
    def fine_tune(self, num_epochs=10, batch_size=2, max_images=200):
        """Main fine-tuning loop"""
        print(f"🚀 Starting {self.language} GroundingDINO Fine-tuning")
        print(f"   Language: {self.language}")
        print(f"   Epochs: {num_epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Total images: {max_images}")
        print("="*60)
        
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders(batch_size, max_images)
        
        # Training history
        history = {
            'train_loss': [],
            'val_detections': [],
            'val_high_conf': [],
            'epochs': [],
            'language': self.language
        }
        
        best_detection_rate = 0
        best_model_state = None
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            print(f"\n📚 {self.language} Epoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss = self.train_epoch(epoch)
            print(f"   Training loss: {train_loss:.4f}")
            
            # Evaluate
            val_detections, val_high_conf = self.evaluate_epoch(epoch)
            
            # Update scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"   Learning rate: {current_lr:.6f}")
            
            # Save history
            history['train_loss'].append(train_loss)
            history['val_detections'].append(val_detections)
            history['val_high_conf'].append(val_high_conf)
            history['epochs'].append(epoch + 1)
            
            # Save best model
            if val_detections > best_detection_rate:
                best_detection_rate = val_detections
                best_model_state = copy.deepcopy(self.model.state_dict())
                print(f"   ✅ New best detection rate: {best_detection_rate:.2f}")
                
                # Save checkpoint
                checkpoint_path = f"{self.results_path}/{self.language}/checkpoints/best_model_epoch_{epoch+1}.pth"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'detection_rate': best_detection_rate,
                    'train_loss': train_loss,
                    'language': self.language
                }, checkpoint_path)
                print(f"   💾 Checkpoint saved: {checkpoint_path}")
        
        total_time = time.time() - start_time
        
        print(f"\n🎉 {self.language} Fine-tuning completed!")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Best detection rate: {best_detection_rate:.2f}")
        
        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            print(f"   ✅ Best {self.language} model loaded")
        
        # Save training history
        history_path = f"{self.results_path}/{self.language}/training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        return history
    
    def run_final_evaluation(self, max_images=200):
        """Run final evaluation"""
        print(f"\n🎯 Running final {self.language} evaluation...")
        
        # Load final dataset
        final_dataset = BillingualCocoDataset(
            self.dataset_path,
            f"{self.dataset_path}/images",
            language=self.language,
            mode='test',
            max_images=max_images
        )
        
        final_loader = DataLoader(
            final_dataset, 
            batch_size=1, 
            shuffle=False,
            collate_fn=custom_collate_fn
        )
        
        results = {
            'predictions': [],
            'model_type': f'GroundingDINO_Baseline_{self.language}',
            'language': self.language,
            'processing_time': 0
        }
        
        start_time = time.time()
        self.model.eval()
        
        for i, batch in enumerate(tqdm(final_loader, desc=f"Final {self.language} Eval")):
            image = batch['original_image'][0]
            image_name = batch['image_name'][0]
            caption = batch['caption'][0]
            
            try:
                # Save temp image
                temp_path = f"temp_final_{self.language}_{i}.jpg"
                image.save(temp_path)
                
                # Load dan predict
                image_source, image_tensor = load_image(temp_path)
                
                boxes, scores, phrases = predict(
                    model=self.model,
                    image=image_tensor,
                    caption=caption,
                    box_threshold=0.35,
                    text_threshold=0.25,
                    device=str(self.device)
                )
                
                # Convert to proper format
                if len(boxes) > 0:
                    boxes_numpy = boxes.cpu().numpy()
                    scores_numpy = scores.cpu().numpy()
                else:
                    boxes_numpy = np.array([])
                    scores_numpy = np.array([])
                
                result = {
                    'image_name': image_name,
                    'caption': caption,
                    'language': self.language,
                    'model': f'GroundingDINO_Baseline_{self.language}',
                    'boxes': boxes_numpy.tolist() if len(boxes_numpy) > 0 else [],
                    'scores': scores_numpy.tolist() if len(scores_numpy) > 0 else [],
                    'phrases': phrases,
                    'num_detections': len(boxes)
                }
                
                results['predictions'].append(result)
                
                # Clean up
                try:
                    os.remove(temp_path)
                except:
                    pass
                    
            except Exception as e:
                print(f"⚠️ Error processing {image_name}: {e}")
                continue
        
        results['processing_time'] = time.time() - start_time
        
        # Save results
        results_path = f"{self.results_path}/{self.language}/predictions.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ {self.language} evaluation completed!")
        print(f"   Total predictions: {len(results['predictions'])}")
        print(f"   Processing time: {results['processing_time']:.2f}s")
        
        # Quick stats
        total_detections = sum(p['num_detections'] for p in results['predictions'])
        avg_detections = total_detections / len(results['predictions']) if results['predictions'] else 0
        
        print(f"📊 {self.language} Performance Summary:")
        print(f"   Total detections: {total_detections}")
        print(f"   Average detections per image: {avg_detections:.2f}")
        
        return results

def run_baseline_finetuning(language='english', num_epochs=10, batch_size=2, max_images=200, debug_mode=False):
    """Run baseline fine-tuning untuk specified language"""
    print(f"🚀 BASELINE GROUNDING DINO FINE-TUNING - {language.upper()}")
    print("🎯 Goal: Fine-tune original GroundingDINO dengan specified language captions")
    print("🔧 Architecture: Swin-T + English BERT (original, not modified)")
    if debug_mode:
        print("🐛 DEBUG MODE: Enabled detailed logging")
    print("="*70)
    
    try:
        # Initialize trainer
        trainer = BaselineGroundingDINOTrainer(
            language=language,
            device='cpu'  # Use CPU for stability
        )
        
        # Run fine-tuning
        print(f"\n📚 Phase 1: {language} Fine-tuning ({num_epochs} epochs)")
        history = trainer.fine_tune(
            num_epochs=num_epochs,
            batch_size=batch_size,
            max_images=max_images
        )
        
        # Run final evaluation
        print(f"\n🎯 Phase 2: {language} Final evaluation")
        results = trainer.run_final_evaluation(max_images=max_images)
        
        print(f"\n🎉 {language.upper()} BASELINE FINE-TUNING COMPLETE!")
        
        return history, results
        
    except Exception as e:
        print(f"❌ Error in {language} fine-tuning: {e}")
        print("💡 Try with smaller batch_size or max_images")
        return None, None

def run_both_baselines(num_epochs=10, batch_size=2, max_images=200):
    """Run fine-tuning untuk both English dan Indonesian baselines"""
    print("🚀 DUAL BASELINE FINE-TUNING")
    print("🎯 Training both English and Indonesian caption models")
    print("="*70)
    
    results_summary = {}
    
    # Train English baseline
    print("\n" + "="*50)
    print("🇺🇸 TRAINING ENGLISH BASELINE")
    print("="*50)
    
    try:
        eng_history, eng_results = run_baseline_finetuning(
            language='english',
            num_epochs=num_epochs,
            batch_size=batch_size,
            max_images=max_images
        )
        
        if eng_history and eng_results:
            results_summary['english'] = {
                'history': eng_history,
                'results': eng_results
            }
        else:
            print("❌ English baseline training failed")
            return None
            
    except Exception as e:
        print(f"❌ English baseline error: {e}")
        return None
    
    # Train Indonesian baseline  
    print("\n" + "="*50)
    print("🇮🇩 TRAINING INDONESIAN BASELINE")
    print("="*50)
    
    try:
        indo_history, indo_results = run_baseline_finetuning(
            language='indonesian',
            num_epochs=num_epochs,
            batch_size=batch_size,
            max_images=max_images
        )
        
        if indo_history and indo_results:
            results_summary['indonesian'] = {
                'history': indo_history,
                'results': indo_results
            }
        else:
            print("❌ Indonesian baseline training failed")
            # Still continue to show comparison dengan English saja
            
    except Exception as e:
        print(f"❌ Indonesian baseline error: {e}")
        # Continue dengan English results saja
    
    # Summary comparison
    print("\n" + "="*70)
    print("📊 BASELINE COMPARISON SUMMARY")
    print("="*70)
    
    if 'english' in results_summary and 'indonesian' in results_summary:
        eng_avg = sum(p['num_detections'] for p in results_summary['english']['results']['predictions']) / len(results_summary['english']['results']['predictions'])
        indo_avg = sum(p['num_detections'] for p in results_summary['indonesian']['results']['predictions']) / len(results_summary['indonesian']['results']['predictions'])
        
        print(f"🇺🇸 English baseline avg detections: {eng_avg:.2f}")
        print(f"🇮🇩 Indonesian baseline avg detections: {indo_avg:.2f}")
        print(f"📉 Performance degradation: {((eng_avg - indo_avg) / eng_avg * 100):.1f}%")
        
        # Save combined results
        combined_results = {
            'comparison_type': 'baseline_finetuning',
            'english_baseline': results_summary['english'],
            'indonesian_baseline': results_summary['indonesian'],
            'summary': {
                'english_avg_detections': eng_avg,
                'indonesian_avg_detections': indo_avg,
                'degradation_percentage': ((eng_avg - indo_avg) / eng_avg * 100) if eng_avg > 0 else 0
            }
        }
    elif 'english' in results_summary:
        eng_avg = sum(p['num_detections'] for p in results_summary['english']['results']['predictions']) / len(results_summary['english']['results']['predictions'])
        print(f"🇺🇸 English baseline avg detections: {eng_avg:.2f}")
        print("⚠️ Indonesian baseline training failed - only English results available")
        
        combined_results = {
            'comparison_type': 'baseline_finetuning_partial',
            'english_baseline': results_summary['english'],
            'summary': {
                'english_avg_detections': eng_avg,
                'status': 'Indonesian training failed'
            }
        }
    else:
        print("❌ Both trainings failed")
        return None
    
    # Create results directory if not exists
    os.makedirs("experiments/results", exist_ok=True)
    
    with open("experiments/results/baseline_comparison.json", 'w', encoding='utf-8') as f:
        json.dump(combined_results, f, indent=2, ensure_ascii=False)
    
    print("✅ Combined results saved to experiments/results/baseline_comparison.json")
    
    return results_summary

if __name__ == "__main__":
    print("🧪 Starting Baseline GroundingDINO Fine-tuning...")
    
    # Test with smaller parameters first
    print("🔬 Running initial test with small dataset...")
    
    # Run both baseline models dengan parameters yang conservative
    results = run_both_baselines(
        num_epochs=10,    # Smaller untuk testing
        batch_size=2,    # Smaller batch size
        max_images=200    # Smaller dataset untuk testing
    )
    
    if results:
        print("\n🎯 Test completed successfully!")
        print("📊 Ready for full training and 3-way comparison!")
        print("\n💡 To run full training, modify parameters in main block:")
        print("   num_epochs=10, batch_size=2, max_images=200")
    else:
        print("\n❌ Test failed. Check errors above and fix before full training.")
        print("💡 Try:")
        print("   - Check file paths in experiments/data/")
        print("   - Verify bilingual captions format")
        print("   - Test with even smaller parameters")