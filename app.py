import os
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from dotenv import load_dotenv
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from PIL import Image

# Import database models
from database import db, User, Item, Match

# --- App Configuration ---
load_dotenv()
app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'instance', 'campus_lostfound.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.path.join(basedir, 'uploads')
os.makedirs(os.path.join(basedir, 'instance'), exist_ok=True)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# --- AI Model Setup ---
model = SentenceTransformer('clip-ViT-B-32')
print("✅ AI Model (CLIP) loaded successfully.")

def get_image_embedding(image_path):
    image = Image.open(image_path)
    with torch.no_grad():
        embedding = model.encode(image, convert_to_tensor=True)
    return embedding.cpu().numpy()

def get_text_embedding(text):
    with torch.no_grad():
        embedding = model.encode(text, convert_to_tensor=True)
    return embedding.cpu().numpy()

# --- User Loader ---
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- Routes ---
@app.route('/')
@login_required
def index():
    if current_user.is_admin:
        all_users = User.query.order_by(User.id).all()
        lost_items = Item.query.filter_by(item_type='lost').order_by(Item.timestamp.desc()).all()
        found_items = Item.query.filter_by(item_type='found').order_by(Item.timestamp.desc()).all()
        return render_template('admin_dashboard.html', users=all_users, lost_items=lost_items, found_items=found_items)
    else:
        # User sees only their ACTIVE items on the main dashboard
        my_lost_items = Item.query.filter_by(user_id=current_user.id, item_type='lost', status='active').order_by(Item.timestamp.desc()).all()
        my_found_items = Item.query.filter_by(user_id=current_user.id, item_type='found', status='active').order_by(Item.timestamp.desc()).all()
        
        # Notification query now joins to get finder's details
        my_lost_item_ids = [item.id for item in my_lost_items]
        notifications = db.session.query(Match, Item, User).join(Item, Match.found_item_id == Item.id).join(User, Item.user_id == User.id).filter(Match.lost_item_id.in_(my_lost_item_ids), Match.status == 'pending').all()

        return render_template('user_dashboard.html', lost_items=my_lost_items, found_items=my_found_items, notifications=notifications)

@app.route('/history')
@login_required
def history():
    if current_user.is_admin:
        return redirect(url_for('index'))

    returned_items = Item.query.filter_by(user_id=current_user.id, status='returned').order_by(Item.timestamp.desc()).all()
    return render_template('history.html', items=returned_items)

def find_matches(new_item, threshold=0.60):
    if new_item.item_type == 'lost':
        comparison_items = Item.query.filter_by(item_type='found', status='active').all()
    else:
        comparison_items = Item.query.filter_by(item_type='lost', status='active').all()
        
    if not comparison_items: return

    new_item_text_emb = np.array(new_item.text_embedding).reshape(1, -1)
    new_item_image_emb = np.array(new_item.image_embedding).reshape(1, -1)
    
    for item in comparison_items:
        if item.text_embedding is None or item.image_embedding is None: continue

        comp_item_text_emb = np.array(item.text_embedding).reshape(1, -1)
        comp_item_image_emb = np.array(item.image_embedding).reshape(1, -1)

        text_sim = cosine_similarity(new_item_text_emb, comp_item_text_emb)[0][0]
        image_sim = cosine_similarity(new_item_image_emb, comp_item_image_emb)[0][0]
        combined_sim = (image_sim * 0.6) + (text_sim * 0.4)
        
        if combined_sim >= threshold:
            lost_item = item if new_item.item_type == 'found' else new_item
            found_item = new_item if new_item.item_type == 'found' else item
            new_match = Match(lost_item_id=lost_item.id, found_item_id=found_item.id, similarity_score=combined_sim)
            db.session.add(new_match)
            
    db.session.commit()

@app.route('/report', methods=['GET', 'POST'])
@login_required
def report():
    if request.method == 'POST':
        image = request.files.get('image')
        if not image or not image.filename:
            flash('Image is required.', 'error')
            return redirect(url_for('report'))

        filename = secure_filename(f"{datetime.utcnow().timestamp()}_{image.filename}")
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(image_path)

        new_item = Item(
            user_id=current_user.id,
            item_type=request.form.get('item_type'),
            title=request.form.get('title'),
            description=request.form.get('description'),
            image_file=filename,
            latitude=float(request.form.get('latitude')) if request.form.get('latitude') else None,
            longitude=float(request.form.get('longitude')) if request.form.get('longitude') else None,
            location_landmark=request.form.get('location_landmark')
        )
        new_item.image_embedding = get_image_embedding(image_path)
        new_item.text_embedding = get_text_embedding(f"{new_item.title} {new_item.description}")
        
        db.session.add(new_item)
        db.session.commit()
        find_matches(new_item, threshold=0.70)

        flash('Your report has been submitted!', 'success')
        return redirect(url_for('index'))
        
    return render_template('report.html')
    
@app.route('/complete_return/<int:match_id>', methods=['POST'])
@login_required
def complete_return(match_id):
    match = Match.query.get_or_404(match_id)
    lost_item = Item.query.get_or_404(match.lost_item_id)
    found_item = Item.query.get_or_404(match.found_item_id)
    finder = User.query.get_or_404(found_item.user_id)

    if lost_item.user_id != current_user.id:
        flash("You are not authorized to perform this action.", "error")
        return redirect(url_for('index'))

    if lost_item.user_id == found_item.user_id:
        flash("Self-reward prevented. Item marked as returned without awarding coins.", "error")
        match.status = 'returned'
        lost_item.status = 'returned'
        found_item.status = 'returned'
        db.session.commit()
        return redirect(url_for('index'))

    match.status = 'returned'
    lost_item.status = 'returned'
    found_item.status = 'returned'
    finder.coins += 100
    db.session.commit()
    
    flash(f"Return confirmed! 100 coins awarded to {finder.username}.", "success")
    return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form.get('username')).first()
        if user and check_password_hash(user.password, request.form.get('password')):
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password.', 'error')
    return render_template('login.html', action='Login')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        phone_number = request.form.get('phone_number')
        
        if User.query.filter_by(username=username).first():
            flash("Username already exists.", "error")
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, password=hashed_password, phone_number=phone_number)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.cli.command("init-db")
def init_db_command():
    with app.app_context():
        db.create_all()
        if not User.query.filter_by(username='admin').first():
            admin_pass = 'admin123'
            hashed_pass = generate_password_hash(admin_pass, method='pbkdf2:sha256')
            admin = User(username='admin', phone_number='N/A', password=hashed_pass, is_admin=True)
            db.session.add(admin)
            db.session.commit()
            print(f"Database initialized and admin user created with password: {admin_pass}")

if __name__ == '__main__':
    app.run(debug=True, port=5012)