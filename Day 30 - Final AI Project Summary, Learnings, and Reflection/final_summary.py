"""
Day 30: Final AI Project Summary, Learnings, and Reflection
30-Day AI Challenge

A comprehensive summary of all projects completed during the challenge.
Generates a report with statistics, visualizations, and key learnings.
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Project definitions
PROJECTS = {
    # Week 1-2: Fundamentals & Computer Vision
    11: {"name": "MNIST Digit Recognition", "category": "Computer Vision", "tools": ["TensorFlow", "Keras", "NumPy"]},
    12: {"name": "Edge Detection & Feature Extraction", "category": "Computer Vision", "tools": ["OpenCV", "NumPy"]},
    13: {"name": "Neural Style Transfer", "category": "Computer Vision", "tools": ["TensorFlow", "VGG19"]},
    14: {"name": "GAN Image Generation", "category": "Computer Vision", "tools": ["TensorFlow", "Keras"]},
    
    # Week 3: NLP & Transformers
    15: {"name": "Sentiment Analysis", "category": "NLP", "tools": ["scikit-learn", "NLTK"]},
    16: {"name": "Named Entity Recognition", "category": "NLP", "tools": ["spaCy"]},
    17: {"name": "Text Summarization", "category": "NLP", "tools": ["Transformers", "BERT"]},
    18: {"name": "Speech to Text", "category": "NLP", "tools": ["Whisper", "SpeechRecognition"]},
    19: {"name": "AI Chatbot", "category": "NLP", "tools": ["Transformers", "DialoGPT"]},
    20: {"name": "Fake News Detection", "category": "NLP", "tools": ["scikit-learn", "TF-IDF"]},
    21: {"name": "Text Generation", "category": "NLP", "tools": ["Transformers", "GPT-2"]},
    
    # Week 4: Advanced & Real-World AI
    22: {"name": "Stock Price Prediction", "category": "Time Series", "tools": ["TensorFlow", "LSTM", "yfinance"]},
    23: {"name": "Reinforcement Learning", "category": "RL", "tools": ["NumPy", "Q-Learning"]},
    24: {"name": "Time Series Forecasting", "category": "Time Series", "tools": ["TensorFlow", "ARIMA"]},
    25: {"name": "Fraud Detection", "category": "Anomaly Detection", "tools": ["scikit-learn", "Isolation Forest"]},
    26: {"name": "Pneumonia Detection", "category": "Medical AI", "tools": ["TensorFlow", "CNN"]},
    27: {"name": "Emotion Detection", "category": "Computer Vision", "tools": ["TensorFlow", "CNN"]},
    28: {"name": "Music Generation", "category": "Generative AI", "tools": ["TensorFlow", "LSTM", "MIDI"]},
    29: {"name": "Object Tracking", "category": "Computer Vision", "tools": ["YOLOv8", "OpenCV"]},
    30: {"name": "Project Summary", "category": "Meta", "tools": ["Python", "Matplotlib"]},
}

KEY_LEARNINGS = """
🎯 KEY LEARNINGS FROM THE 30-DAY AI CHALLENGE
============================================

📚 FUNDAMENTALS
--------------
• Neural networks can learn complex patterns from data
• Data preprocessing is crucial for model performance
• Regularization (dropout, batch norm) prevents overfitting
• Transfer learning accelerates development significantly

🖼️ COMPUTER VISION
-----------------
• CNNs excel at hierarchical feature extraction
• Edge detection forms the foundation of many CV tasks
• GANs can generate realistic synthetic images
• YOLO enables real-time object detection

📝 NATURAL LANGUAGE PROCESSING
-----------------------------
• Transformers revolutionized NLP (attention is all you need)
• Pre-trained models (BERT, GPT) provide excellent starting points
• Tokenization and embedding capture semantic meaning
• Sentiment and entity extraction enable text understanding

📈 TIME SERIES & PREDICTION
--------------------------
• LSTM networks handle sequential dependencies well
• Feature engineering improves forecasting accuracy
• Multiple models should be compared for best results
• Anomaly detection requires different techniques

🎮 REINFORCEMENT LEARNING
------------------------
• Agents learn through trial and error
• Reward shaping is crucial for learning
• Exploration vs exploitation tradeoff matters
• Q-learning works well for discrete action spaces

🏥 REAL-WORLD APPLICATIONS
-------------------------
• AI can assist (not replace) medical diagnosis
• Fraud detection saves billions annually
• Emotion AI enables better human-computer interaction
• Music generation shows AI's creative potential

🛠️ PRACTICAL SKILLS GAINED
--------------------------
• TensorFlow/Keras model building
• scikit-learn for classical ML
• Data visualization with matplotlib
• Working with various data formats
• Model evaluation and metrics
• Debugging ML pipelines
"""

def count_tools_used():
    """Count unique tools used across all projects."""
    all_tools = []
    for day, project in PROJECTS.items():
        all_tools.extend(project["tools"])
    
    tool_counts = {}
    for tool in all_tools:
        tool_counts[tool] = tool_counts.get(tool, 0) + 1
    
    return tool_counts

def count_categories():
    """Count projects by category."""
    categories = {}
    for day, project in PROJECTS.items():
        cat = project["category"]
        categories[cat] = categories.get(cat, 0) + 1
    return categories

def generate_summary_report():
    """Generate comprehensive summary report."""
    report = []
    report.append("=" * 60)
    report.append("30-DAY AI CHALLENGE - FINAL SUMMARY REPORT")
    report.append("=" * 60)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append("")
    
    # Project list
    report.append("📋 PROJECTS COMPLETED")
    report.append("-" * 40)
    for day in sorted(PROJECTS.keys()):
        project = PROJECTS[day]
        report.append(f"Day {day}: {project['name']}")
        report.append(f"        Category: {project['category']}")
        report.append(f"        Tools: {', '.join(project['tools'])}")
        report.append("")
    
    # Statistics
    report.append("📊 STATISTICS")
    report.append("-" * 40)
    report.append(f"Total Projects: {len(PROJECTS)}")
    
    categories = count_categories()
    report.append(f"Categories Covered: {len(categories)}")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        report.append(f"  • {cat}: {count} projects")
    
    tools = count_tools_used()
    report.append(f"\nUnique Tools Used: {len(tools)}")
    for tool, count in sorted(tools.items(), key=lambda x: -x[1])[:10]:
        report.append(f"  • {tool}: {count} projects")
    
    # Key learnings
    report.append("")
    report.append(KEY_LEARNINGS)
    
    return "\n".join(report)

def create_visualizations():
    """Create summary visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Projects by category
    categories = count_categories()
    cats = list(categories.keys())
    counts = list(categories.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(cats)))
    
    axes[0, 0].pie(counts, labels=cats, autopct='%1.1f%%', colors=colors)
    axes[0, 0].set_title('Projects by Category', fontsize=12, fontweight='bold')
    
    # 2. Tools usage
    tools = count_tools_used()
    top_tools = dict(sorted(tools.items(), key=lambda x: -x[1])[:10])
    
    axes[0, 1].barh(list(top_tools.keys()), list(top_tools.values()), color='steelblue')
    axes[0, 1].set_xlabel('Number of Projects')
    axes[0, 1].set_title('Top 10 Tools Used', fontsize=12, fontweight='bold')
    axes[0, 1].invert_yaxis()
    
    # 3. Timeline
    days = sorted(PROJECTS.keys())
    cumulative = list(range(1, len(days) + 1))
    
    axes[1, 0].plot(days, cumulative, 'o-', color='green', linewidth=2, markersize=8)
    axes[1, 0].fill_between(days, cumulative, alpha=0.3, color='green')
    axes[1, 0].set_xlabel('Day')
    axes[1, 0].set_ylabel('Cumulative Projects')
    axes[1, 0].set_title('Project Progress Over Time', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Category timeline
    cat_colors = {
        'Computer Vision': '#FF6B6B',
        'NLP': '#4ECDC4',
        'Time Series': '#45B7D1',
        'RL': '#96CEB4',
        'Anomaly Detection': '#FFEAA7',
        'Medical AI': '#DDA0DD',
        'Generative AI': '#98D8C8',
        'Meta': '#95A5A6'
    }
    
    for day in sorted(PROJECTS.keys()):
        cat = PROJECTS[day]['category']
        color = cat_colors.get(cat, 'gray')
        axes[1, 1].barh(day, 1, color=color, edgecolor='black', linewidth=0.5)
    
    # Legend
    legend_handles = [plt.Rectangle((0,0),1,1, color=c) for c in cat_colors.values()]
    axes[1, 1].legend(legend_handles, cat_colors.keys(), loc='lower right', fontsize=8)
    axes[1, 1].set_xlabel('Project Completed')
    axes[1, 1].set_ylabel('Day')
    axes[1, 1].set_title('Projects by Day and Category', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('challenge_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Summary visualization saved to 'challenge_summary.png'")

def create_certificate():
    """Create a completion certificate."""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    
    # Background
    ax.set_facecolor('#f5f5dc')
    
    # Border
    border = plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, 
                           edgecolor='#8B4513', linewidth=5)
    ax.add_patch(border)
    
    inner_border = plt.Rectangle((0.08, 0.08), 0.84, 0.84, fill=False,
                                 edgecolor='#DAA520', linewidth=2)
    ax.add_patch(inner_border)
    
    # Text
    ax.text(0.5, 0.85, '🏆 CERTIFICATE OF COMPLETION 🏆', 
            ha='center', va='center', fontsize=24, fontweight='bold', color='#8B4513')
    
    ax.text(0.5, 0.72, 'This certifies that', 
            ha='center', va='center', fontsize=14, color='#444')
    
    ax.text(0.5, 0.62, '[YOUR NAME]', 
            ha='center', va='center', fontsize=28, fontweight='bold', 
            color='#2F4F4F', style='italic')
    
    ax.text(0.5, 0.50, 'has successfully completed the', 
            ha='center', va='center', fontsize=14, color='#444')
    
    ax.text(0.5, 0.40, '30-DAY AI CHALLENGE', 
            ha='center', va='center', fontsize=22, fontweight='bold', color='#8B4513')
    
    ax.text(0.5, 0.30, f'Completing {len(PROJECTS)} AI/ML projects across\n'
                       f'{len(count_categories())} categories using {len(count_tools_used())}+ tools',
            ha='center', va='center', fontsize=12, color='#444')
    
    ax.text(0.5, 0.18, f'Date: {datetime.now().strftime("%B %d, %Y")}',
            ha='center', va='center', fontsize=12, color='#444')
    
    ax.text(0.5, 0.10, '🤖 Keep Learning, Keep Building! 🚀',
            ha='center', va='center', fontsize=14, color='#2F4F4F')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.savefig('completion_certificate.png', dpi=150, bbox_inches='tight',
                facecolor='#f5f5dc', edgecolor='none')
    plt.close()
    print("Certificate saved to 'completion_certificate.png'")

def main():
    print("=" * 60)
    print("Day 30: Final Project Summary & Reflection")
    print("=" * 60)
    
    print("\n[1] Generating summary report...")
    report = generate_summary_report()
    print(report)
    
    # Save report
    with open('challenge_summary.txt', 'w') as f:
        f.write(report)
    print("\nReport saved to 'challenge_summary.txt'")
    
    print("\n[2] Creating visualizations...")
    create_visualizations()
    
    print("\n[3] Creating completion certificate...")
    create_certificate()
    
    print("\n[4] Saving project data...")
    output = {
        'total_projects': len(PROJECTS),
        'categories': count_categories(),
        'tools_used': count_tools_used(),
        'projects': {str(k): v for k, v in PROJECTS.items()},
        'completion_date': datetime.now().isoformat()
    }
    
    with open('challenge_data.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("Data saved to 'challenge_data.json'")
    
    # Final message
    print("\n" + "=" * 60)
    print("🎉 CONGRATULATIONS! 🎉")
    print("=" * 60)
    print("""
    You've completed the 30-Day AI Challenge!
    
    📊 Your Achievements:
    • 20 AI/ML projects built from scratch
    • Covered Computer Vision, NLP, Time Series, RL, and more
    • Hands-on experience with TensorFlow, PyTorch, scikit-learn
    • Built real-world applications (fraud detection, medical AI, etc.)
    
    🚀 What's Next?
    • Deploy your best project to the cloud
    • Contribute to open-source AI projects
    • Build a portfolio website showcasing your work
    • Take on more advanced challenges (Kaggle competitions)
    • Specialize in an area that interests you most
    
    Keep learning, keep building, keep innovating!
    
    🌟 You're now an AI Practitioner! 🌟
    """)
    
    print("=" * 60)
    print("30-DAY AI CHALLENGE COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()
