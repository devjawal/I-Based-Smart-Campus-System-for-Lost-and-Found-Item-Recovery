# A Multimodal AI-Based Smart Campus System for Lost and Found Item Recovery

This project is a web-based Smart Campus Lost and Found System (SCLFS) developed as part of academic coursework at Vellore Institute of Technology (VIT).

## 📍 Overview

Traditional lost-and-found systems on busy university campuses are often slow and inefficient. This project solves that problem by providing a modern web application that uses artificial intelligence to intelligently match lost items with found items. It's built on a lightweight Flask and SQLAlchemy backend.

## ✨ Unique Features

This system's innovation lies in its synthesis of two key components:

1.  **🧠 Multimodal AI Matching Engine:**
    * Uses the **CLIP (ViT-B/32) vision-language model** to understand both text and images.
    * Instead of a simple keyword search, it calculates a *composite similarity score* by comparing an item's description (e.g., "my black steel water bottle") with a found item's uploaded image.
    * This allows the system to find matches with high accuracy, even if the descriptions aren't perfect.

2.  **🏆 Socio-Technical Incentive Framework (Gamification):**
    * A system is only as good as its user participation. To encourage students to report found items, the platform features a gamified reward system.
    * Users who successfully return an item to its owner receive **virtual tokens (coins)**.
    * This model is designed to promote ethical participation and build a helpful campus community.

## 🛠️ Tech Stack

* **Backend:** Python, Flask
* **Database:** SQLAlchemy (ORM)
* **AI/ML:** sentence-transformers, CLIP

## 🧑‍💻 Project Team

### Authors
* Devkaran Jawal (22BCE3048)
* Nikhil Singla (22BKT0015)
* Joshua Roland Williams (22BCE3022)
* Sumit Kumar (22BDS0166)

### Faculty Supervisor
* Swarnalatha P

This project was based on research outlined in the paper "A Multimodal AI-Based Smart Campus System for Lost and Found Item Recovery."