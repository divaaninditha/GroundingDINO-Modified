"""
Prepare 200 COCO images with multiple objects for GroundingDINO experiments
Filter gambar dengan banyak objek, translate captions dengan Helsinki-NLP, dan copy ke folder eksperimen
"""

import json
import os
import shutil
from collections import defaultdict, Counter
import random
from PIL import Image

# Install Helsinki-NLP translation if not available
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch
    TRANSLATION_AVAILABLE = True
except ImportError:
    print("⚠️  Helsinki-NLP translation not available. Run: pip install transformers")
    TRANSLATION_AVAILABLE = False

# COCO dataset paths
COCO_ROOT = "D:/datasets/coco2017"
COCO_IMAGES = os.path.join(COCO_ROOT, "val2017")  # atau train2017
COCO_ANNOTATIONS = os.path.join(COCO_ROOT, "annotations/instances_val2017.json")  # atau instances_train2017.json
COCO_CAPTIONS = os.path.join(COCO_ROOT, "annotations/captions_val2017.json")  # COCO captions file

# Target paths
OUTPUT_ROOT = "experiments/data"
OUTPUT_IMAGES = os.path.join(OUTPUT_ROOT, "images")
OUTPUT_ANNOTATIONS = os.path.join(OUTPUT_ROOT, "annotations.json")
OUTPUT_CAPTIONS = os.path.join(OUTPUT_ROOT, "captions")

# Translation mapping English -> Indonesian
CAPTION_TRANSLATION = {
    # Person & Body
    "person": "orang",
    "people": "orang-orang",
    
    # Vehicles
    "car": "mobil", 
    "truck": "truk",
    "bus": "bus",
    "motorcycle": "motor",
    "bicycle": "sepeda",
    "airplane": "pesawat",
    "boat": "perahu",
    "train": "kereta",
    
    # Animals
    "bird": "burung",
    "cat": "kucing", 
    "dog": "anjing",
    "horse": "kuda",
    "sheep": "domba",
    "cow": "sapi",
    "elephant": "gajah",
    "bear": "beruang",
    "zebra": "zebra",
    "giraffe": "jerapah",
    
    # Food & Kitchen
    "banana": "pisang",
    "apple": "apel",
    "sandwich": "roti lapis",
    "orange": "jeruk",
    "broccoli": "brokoli",
    "carrot": "wortel",
    "pizza": "pizza",
    "donut": "donat",
    "cake": "kue",
    "chair": "kursi",
    "dining table": "meja makan",
    "cup": "cangkir",
    "fork": "garpu",
    "knife": "pisau",
    "spoon": "sendok",
    "bowl": "mangkuk",
    "wine glass": "gelas wine",
    "bottle": "botol",
    
    # Electronics & Objects
    "laptop": "laptop",
    "tv": "tv",
    "remote": "remote",
    "keyboard": "keyboard",
    "cell phone": "hp",
    "microwave": "microwave",
    "oven": "oven",
    "toaster": "pemanggang",
    "sink": "wastafel",
    "refrigerator": "kulkas",
    "clock": "jam",
    
    # Furniture & Indoor
    "couch": "sofa",
    "bed": "tempat tidur",
    "toilet": "toilet",
    "book": "buku",
    "vase": "vas",
    "scissors": "gunting",
    "teddy bear": "boneka beruang",
    "hair drier": "pengering rambut",
    "toothbrush": "sikat gigi",
    
    # Sports & Outdoor
    "frisbee": "frisbee",
    "skis": "ski",
    "snowboard": "snowboard",
    "sports ball": "bola",
    "kite": "layang-layang",
    "baseball bat": "tongkat baseball",
    "baseball glove": "sarung tangan baseball",
    "skateboard": "skateboard",
    "surfboard": "papan selancar",
    "tennis racket": "raket tenis",
    
    # Bags & Accessories  
    "backpack": "tas ransel",
    "handbag": "tas tangan",
    "suitcase": "koper",
    "umbrella": "payung",
    "tie": "dasi",
    
    # Traffic & Street
    "traffic light": "lampu lalu lintas",
    "fire hydrant": "hidran",
    "stop sign": "rambu stop",
    "parking meter": "meteran parkir",
    "bench": "bangku"
}

def load_coco_annotations():
    """Load COCO annotations"""
    print("📖 Loading COCO annotations...")
    with open(COCO_ANNOTATIONS, 'r') as f:
        coco_data = json.load(f)
    
    print(f"✅ Loaded {len(coco_data['images'])} images")
    print(f"✅ Loaded {len(coco_data['annotations'])} annotations") 
    print(f"✅ Loaded {len(coco_data['categories'])} categories")
    
    return coco_data

def load_coco_captions():
    """Load COCO captions"""
    print("📝 Loading COCO captions...")
    
    if not os.path.exists(COCO_CAPTIONS):
        print(f"⚠️  COCO captions not found: {COCO_CAPTIONS}")
        return None
    
    with open(COCO_CAPTIONS, 'r') as f:
        captions_data = json.load(f)
    
    # Group captions by image_id
    image_captions = defaultdict(list)
    for caption in captions_data['annotations']:
        image_captions[caption['image_id']].append(caption['caption'])
    
    print(f"✅ Loaded captions for {len(image_captions)} images")
    print(f"✅ Total captions: {len(captions_data['annotations'])}")
    
    return image_captions

def setup_helsinki_translator():
    """Setup Helsinki-NLP English to Indonesian translator"""
    print("🌐 Setting up Helsinki-NLP EN->ID translator...")
    
    if not TRANSLATION_AVAILABLE:
        print("❌ Transformers not available. Please install: pip install transformers")
        return None, None
    
    try:
        model_name = "Helsinki-NLP/opus-mt-en-id"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        print("✅ Helsinki-NLP translator loaded successfully")
        return tokenizer, model
    except Exception as e:
        print(f"❌ Failed to load Helsinki-NLP translator: {e}")
        print("   Will use manual translations only")
        return None, None

def translate_text(text, tokenizer, model):
    """Translate English text to Indonesian using Helsinki-NLP"""
    if not tokenizer or not model:
        return text  # Return original if translator not available
    
    try:
        # Tokenize and translate
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)
        
        # Decode translation
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translation.strip()
    except Exception as e:
        print(f"⚠️  Translation failed for '{text[:50]}...': {e}")
        return text  # Return original if translation fails

def translate_captions_for_images(selected_images, image_captions, tokenizer, model):
    """Translate COCO captions for selected images"""
    print(f"\n🌐 Translating captions for {len(selected_images)} images...")
    
    image_bilingual_captions = {}
    translation_count = 0
    
    for i, img_data in enumerate(selected_images):
        img_id = img_data['image_info']['id']
        img_name = img_data['image_info']['file_name']
        
        # Get captions for this image
        captions = image_captions.get(img_id, [])
        
        if captions:
            english_captions = captions[:3]  # Take first 3 captions per image
            indonesian_captions = []
            
            # Translate each caption
            for caption in english_captions:
                if tokenizer and model:
                    indo_caption = translate_text(caption, tokenizer, model)
                    indonesian_captions.append(indo_caption)
                    translation_count += 1
                else:
                    # Fallback: simple word replacement
                    indo_caption = caption
                    for eng, indo in CAPTION_TRANSLATION.items():
                        indo_caption = indo_caption.replace(eng, indo)
                    indonesian_captions.append(indo_caption)
            
            image_bilingual_captions[img_name] = {
                "english": english_captions,
                "indonesian": indonesian_captions,
                "image_id": img_id
            }
            
            if (i + 1) % 25 == 0:
                print(f"   Processed {i + 1}/{len(selected_images)} images...")
        else:
            print(f"⚠️  No captions found for {img_name}")
    
    print(f"✅ Translated {translation_count} captions")
    print(f"✅ Processed {len(image_bilingual_captions)} images with captions")
    
    return image_bilingual_captions

def analyze_images_by_object_count(coco_data):
    """Analyze images by number of objects"""
    print("\n📊 Analyzing images by object count...")
    
    # Group annotations by image_id
    image_objects = defaultdict(list)
    for ann in coco_data['annotations']:
        if ann['iscrowd'] == 0:  # Skip crowd annotations
            image_objects[ann['image_id']].append(ann)
    
    # Count objects per image
    object_counts = {}
    category_counts = defaultdict(int)
    
    for img in coco_data['images']:
        img_id = img['id']
        objects = image_objects.get(img_id, [])
        object_counts[img_id] = len(objects)
        
        # Count categories
        for obj in objects:
            category_counts[obj['category_id']] += 1
    
    # Statistics
    counts_distribution = Counter(object_counts.values())
    
    print(f"📈 Object count distribution:")
    for count in sorted(counts_distribution.keys())[:10]:
        print(f"   {count} objects: {counts_distribution[count]} images")
    
    return image_objects, object_counts

def select_diverse_images(coco_data, image_objects, object_counts, target_count=200, min_objects=3):
    """Select diverse images with multiple objects"""
    print(f"\n🎯 Selecting {target_count} diverse images (min {min_objects} objects)...")
    
    # Create category mapping
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
    cat_name_to_id = {cat['name']: cat['id'] for cat in coco_data['categories']}
    
    # Filter images with enough objects
    candidate_images = []
    for img in coco_data['images']:
        img_id = img['id']
        if object_counts.get(img_id, 0) >= min_objects:
            # Get categories in this image
            objects = image_objects.get(img_id, [])
            categories = [cat_id_to_name[obj['category_id']] for obj in objects]
            
            candidate_images.append({
                'image_info': img,
                'object_count': len(objects),
                'categories': set(categories),
                'objects': objects
            })
    
    print(f"📋 Found {len(candidate_images)} candidate images")
    
    # Sort by object count (prioritize images with more objects)
    candidate_images.sort(key=lambda x: x['object_count'], reverse=True)
    
    # Select diverse set
    selected_images = []
    selected_categories = defaultdict(int)
    
    # Prioritize images with many objects and diverse categories
    for candidate in candidate_images:
        if len(selected_images) >= target_count:
            break
            
        # Check if this adds diversity
        new_categories = candidate['categories'] - set(selected_categories.keys())
        
        # Select if: few selected OR adds new categories OR high object count
        if (len(selected_images) < 50 or  # Always select first 50
            len(new_categories) > 0 or      # Adds new categories
            candidate['object_count'] >= 8): # High object count
            
            selected_images.append(candidate)
            for cat in candidate['categories']:
                selected_categories[cat] += 1
    
    print(f"✅ Selected {len(selected_images)} images")
    print(f"📊 Category distribution:")
    for cat, count in sorted(selected_categories.items())[:15]:
        indo_name = CAPTION_TRANSLATION.get(cat, cat)
        print(f"   {cat} ({indo_name}): {count}")
    
    return selected_images

def copy_images_and_create_annotations(selected_images, coco_data):
    """Copy selected images and create new annotations"""
    print(f"\n📁 Creating output directories...")
    
    # Create directories
    os.makedirs(OUTPUT_IMAGES, exist_ok=True)
    os.makedirs(OUTPUT_CAPTIONS, exist_ok=True)
    
    # Create category mapping
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # New annotations structure
    new_annotations = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Copy categories with translation
    category_mapping = {}
    for cat in coco_data['categories']:
        new_cat = {
            "id": cat['id'],
            "name": cat['name'],
            "supercategory": cat['supercategory'],
            "name_indonesian": CAPTION_TRANSLATION.get(cat['name'], cat['name'])
        }
        new_annotations["categories"].append(new_cat)
        category_mapping[cat['id']] = cat['name']
    
    # Process selected images
    print(f"📸 Copying {len(selected_images)} images...")
    
    copied_count = 0
    for i, img_data in enumerate(selected_images):
        img_info = img_data['image_info']
        source_path = os.path.join(COCO_IMAGES, img_info['file_name'])
        target_path = os.path.join(OUTPUT_IMAGES, img_info['file_name'])
        
        # Copy image if exists
        if os.path.exists(source_path):
            try:
                shutil.copy2(source_path, target_path)
                copied_count += 1
                
                # Add image info
                new_annotations["images"].append(img_info)
                
                # Add annotations for this image
                for obj in img_data['objects']:
                    new_annotations["annotations"].append(obj)
                
                if (i + 1) % 50 == 0:
                    print(f"   Copied {i + 1}/{len(selected_images)} images...")
                    
            except Exception as e:
                print(f"❌ Error copying {img_info['file_name']}: {e}")
        else:
            print(f"⚠️  Image not found: {source_path}")
    
    print(f"✅ Successfully copied {copied_count} images")
    
    # Save annotations
    with open(OUTPUT_ANNOTATIONS, 'w') as f:
        json.dump(new_annotations, f, indent=2)
    print(f"✅ Saved annotations to {OUTPUT_ANNOTATIONS}")
    
    return new_annotations

def create_caption_mappings(image_bilingual_captions=None):
    """Create comprehensive caption mapping files"""
    print(f"\n📝 Creating comprehensive caption mappings...")
    
    # 1. Basic class name translations
    english_captions = {name: name for name in CAPTION_TRANSLATION.keys()}
    indonesian_captions = CAPTION_TRANSLATION.copy()
    
    # 2. Complex multi-object queries
    complex_queries = {
        # Basic combinations
        "person with bag": "orang dengan tas",
        "person holding phone": "orang memegang hp",
        "person using laptop": "orang menggunakan laptop",
        "person sitting on chair": "orang duduk di kursi",
        "person riding motorcycle": "orang naik motor",
        "person driving car": "orang mengendarai mobil",
        
        # Multiple objects
        "car and motorcycle": "mobil dan motor",
        "car and bicycle": "mobil dan sepeda", 
        "laptop and phone": "laptop dan hp",
        "cup and bottle": "cangkir dan botol",
        "chair and table": "kursi dan meja",
        "cat and dog": "kucing dan anjing",
        
        # Location-based
        "food on table": "makanan di meja",
        "laptop on table": "laptop di meja",
        "bag on chair": "tas di kursi",
        "phone on table": "hp di meja",
        "book on bed": "buku di tempat tidur",
        "remote on couch": "remote di sofa",
        
        # Action-based
        "person eating food": "orang makan makanan",
        "person drinking": "orang minum",
        "person reading book": "orang membaca buku",
        "person watching tv": "orang menonton tv",
        
        # Color/attribute descriptions
        "red car": "mobil merah",
        "black motorcycle": "motor hitam",
        "white laptop": "laptop putih",
        "blue chair": "kursi biru",
        "brown bag": "tas coklat",
        
        # Quantity-based
        "two people": "dua orang",
        "multiple cars": "beberapa mobil",
        "many chairs": "banyak kursi",
        "several bottles": "beberapa botol",
        
        # Indonesian-specific objects
        "warung": "warung",
        "ojek": "ojek", 
        "gorengan": "gorengan",
        "nasi": "nasi",
        "sepatu": "sepatu",
        "helm": "helm"
    }
    
    # 3. Natural language queries
    natural_queries = {
        "a person sitting on a chair": "seseorang duduk di kursi",
        "a car parked on the street": "mobil diparkir di jalan", 
        "a laptop computer on the desk": "komputer laptop di meja",
        "a phone in someone's hand": "hp di tangan seseorang",
        "food served on a plate": "makanan disajikan di piring",
        "a bag hanging on the chair": "tas tergantung di kursi",
        "two people talking": "dua orang berbicara",
        "a cat sleeping on the bed": "kucing tidur di tempat tidur",
        "a bottle of water": "botol air",
        "a cup of coffee": "secangkir kopi"
    }
    
    # 4. Add all translations
    for eng, indo in complex_queries.items():
        english_captions[eng] = eng
        indonesian_captions[eng] = indo
    
    for eng, indo in natural_queries.items():
        english_captions[eng] = eng
        indonesian_captions[eng] = indo
    
    # 5. Save basic mappings
    with open(os.path.join(OUTPUT_CAPTIONS, "class_mappings_english.json"), 'w') as f:
        json.dump(english_captions, f, indent=2)
    
    with open(os.path.join(OUTPUT_CAPTIONS, "class_mappings_indonesian.json"), 'w') as f:
        json.dump(indonesian_captions, f, indent=2, ensure_ascii=False)
    
    # 6. Save COCO bilingual captions if available
    if image_bilingual_captions:
        with open(os.path.join(OUTPUT_CAPTIONS, "coco_captions_bilingual.json"), 'w') as f:
            json.dump(image_bilingual_captions, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved bilingual COCO captions for {len(image_bilingual_captions)} images")
    
    # 7. Create test queries file for experiments
    test_queries = {
        "single_objects": list(CAPTION_TRANSLATION.keys())[:20],
        "complex_queries": list(complex_queries.keys())[:15],
        "natural_queries": list(natural_queries.keys())[:10]
    }
    
    with open(os.path.join(OUTPUT_CAPTIONS, "test_queries.json"), 'w') as f:
        json.dump(test_queries, f, indent=2)
    
    print(f"✅ Saved comprehensive caption mappings")
    print(f"   📝 Basic translations: {len(CAPTION_TRANSLATION)}")
    print(f"   🔀 Complex queries: {len(complex_queries)}")
    print(f"   💬 Natural queries: {len(natural_queries)}")
    print(f"   📊 Total English: {len(english_captions)}")
    print(f"   📊 Total Indonesian: {len(indonesian_captions)}")
    print(f"   🧪 Test queries saved for experiments")

def main():
    """Main function"""
    print("🚀 COCO Dataset Preparation for GroundingDINO Experiments")
    print("🌐 With Helsinki-NLP Caption Translation")
    print("=" * 60)
    
    # Check source paths
    if not os.path.exists(COCO_IMAGES):
        print(f"❌ COCO images not found: {COCO_IMAGES}")
        print("   Please check the path or update COCO_IMAGES variable")
        return
    
    if not os.path.exists(COCO_ANNOTATIONS):
        print(f"❌ COCO annotations not found: {COCO_ANNOTATIONS}")
        print("   Please check the path or update COCO_ANNOTATIONS variable")
        return
    
    # Step 1: Load data
    coco_data = load_coco_annotations()
    image_captions = load_coco_captions()
    
    # Step 2: Setup translation
    tokenizer, model = setup_helsinki_translator()
    
    # Step 3: Analyze images
    image_objects, object_counts = analyze_images_by_object_count(coco_data)
    
    # Step 4: Select diverse images
    selected_images = select_diverse_images(coco_data, image_objects, object_counts)
    
    # Step 5: Copy images and create annotations
    new_annotations = copy_images_and_create_annotations(selected_images, coco_data)
    
    # Step 6: Translate COCO captions for selected images
    image_bilingual_captions = None
    if image_captions:
        image_bilingual_captions = translate_captions_for_images(
            selected_images, image_captions, tokenizer, model
        )
    else:
        print("⚠️  Skipping caption translation (COCO captions not found)")
    
    # Step 7: Create all caption mappings
    create_caption_mappings(image_bilingual_captions)
    
    print("\n" + "=" * 60)
    print("🎉 Dataset preparation completed!")
    print(f"📁 Images: {OUTPUT_IMAGES}")
    print(f"📄 Annotations: {OUTPUT_ANNOTATIONS}")
    print(f"🗣️  Captions: {OUTPUT_CAPTIONS}")
    print(f"📊 Total images: {len(new_annotations['images'])}")
    print(f"📊 Total objects: {len(new_annotations['annotations'])}")
    
    if image_bilingual_captions:
        print(f"🌐 Bilingual captions: {len(image_bilingual_captions)} images")
        print("🎯 Ready for bilingual GroundingDINO experiments!")
        
        # Show example translations
        print(f"\n📝 Example caption translations:")
        sample_img = list(image_bilingual_captions.keys())[0]
        sample_captions = image_bilingual_captions[sample_img]
        print(f"   Image: {sample_img}")
        print(f"   English: {sample_captions['english'][0]}")
        print(f"   Indonesian: {sample_captions['indonesian'][0]}")
    
    print("🚀 Ready for GroundingDINO experiments!")

if __name__ == "__main__":
    main()