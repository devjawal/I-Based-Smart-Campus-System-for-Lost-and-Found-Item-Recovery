import torch
from sentence_transformers import SentenceTransformer
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from database import db, Item, Match

# Load the AI model once when the application starts
# CLIP can encode both images and text into the same vector space
model = SentenceTransformer('clip-ViT-B-32')
print("✅ AI Model (CLIP) loaded successfully.")

def get_image_embedding(image_path):
    """Generates a vector embedding from an image file."""
    try:
        image = Image.open(image_path)
        with torch.no_grad():
            embedding = model.encode(image, convert_to_tensor=True)
        return embedding.cpu().numpy()
    except Exception as e:
        print(f"Error generating image embedding: {e}")
        return None

def get_text_embedding(text):
    """Generates a vector embedding from a text string."""
    try:
        with torch.no_grad():
            embedding = model.encode(text, convert_to_tensor=True)
        return embedding.cpu().numpy()
    except Exception as e:
        print(f"Error generating text embedding: {e}")
        return None

# --- MODIFIED FUNCTION ---
# ai_matching.py

def find_matches(new_item, threshold=0.60):
    """Finds matches for a new item against existing items in the database."""
    # DEBUG: Announce that the function has started
    print("\n--- AI Matching Triggered ---")
    print(f"-> Matching for new '{new_item.item_type}' item #{new_item.id}: '{new_item.title}'")

    if new_item.item_type == 'lost':
        comparison_items = Item.query.filter_by(item_type='found', status='active').all()
    else: # item_type is 'found'
        comparison_items = Item.query.filter_by(item_type='lost', status='active').all()

    # DEBUG: Show how many potential matches were found in the database
    print(f"Found {len(comparison_items)} potential items to compare against.")

    if not comparison_items:
        print("--- AI Matching Finished (No items to compare) ---")
        return

    new_item_text_emb = np.array(new_item.text_embedding).reshape(1, -1)
    new_item_image_emb = np.array(new_item.image_embedding).reshape(1, -1)

    for item in comparison_items:
        if item.text_embedding is None or item.image_embedding is None:
            continue

        comp_item_text_emb = np.array(item.text_embedding).reshape(1, -1)
        comp_item_image_emb = np.array(item.image_embedding).reshape(1, -1)

        text_sim = cosine_similarity(new_item_text_emb, comp_item_text_emb)[0][0]
        image_sim = cosine_similarity(new_item_image_emb, comp_item_image_emb)[0][0]
        combined_sim = (image_sim * 0.6) + (text_sim * 0.4)
        
        # DEBUG: Print the score for every single comparison
        print(f"  - Comparing with item #{item.id}: Img Sim={image_sim:.2f}, Txt Sim={text_sim:.2f} -> Combined Score: {combined_sim:.4f}")

        if combined_sim >= threshold:
            # DEBUG: Announce when a match is successful
            print(f"    ✅ SUCCESS: Match found! Score is >= {threshold}. Creating match record.")
            
            lost_item = item if new_item.item_type == 'found' else new_item
            found_item = new_item if new_item.item_type == 'found' else item
            
            new_match = Match(
                lost_item_id=lost_item.id,
                found_item_id=found_item.id,
                similarity_score=combined_sim
            )
            db.session.add(new_match)
            
    db.session.commit()
    print("--- AI Matching Finished ---")
    """Finds matches for a new item against existing items in the database."""
    if new_item.item_type == 'lost':
        # Compare a new 'lost' item against all 'found' items
        comparison_items = Item.query.filter_by(item_type='found', status='active').all()
    else: # item_type is 'found'
        # Compare a new 'found' item against all 'lost' items
        comparison_items = Item.query.filter_by(item_type='lost', status='active').all()

    if not comparison_items:
        print("No items to compare against.")
        return

    new_item_text_emb = np.array(new_item.text_embedding).reshape(1, -1)
    new_item_image_emb = np.array(new_item.image_embedding).reshape(1, -1)

    for item in comparison_items:
        if item.text_embedding is None or item.image_embedding is None:
            continue

        comp_item_text_emb = np.array(item.text_embedding).reshape(1, -1)
        comp_item_image_emb = np.array(item.image_embedding).reshape(1, -1)

        # Calculate similarities
        text_sim = cosine_similarity(new_item_text_emb, comp_item_text_emb)[0][0]
        image_sim = cosine_similarity(new_item_image_emb, comp_item_image_emb)[0][0]

        # Combine similarities (e.g., 60% image, 40% text)
        combined_sim = (image_sim * 0.6) + (text_sim * 0.4)

        # CHANGED: Use the threshold variable instead of a hardcoded value
        if combined_sim >= threshold:
            print(f"✅ Found a potential match! Score: {combined_sim:.2f}")

            lost_item = item if new_item.item_type == 'found' else new_item
            found_item = new_item if new_item.item_type == 'found' else item

            # Create a new match record
            new_match = Match(
                lost_item_id=lost_item.id,
                found_item_id=found_item.id,
                similarity_score=combined_sim
            )
            db.session.add(new_match)

    db.session.commit()