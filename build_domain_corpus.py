import re
import wikipediaapi

def clean_text(text):
    """Removes Wikipedia reference numbers, headers, and excessive empty lines."""
    # Remove citation brackets like [1], [12]
    text = re.sub(r'\[\d+\]', '', text)
    # Remove section headings like == History == or === Early Years ===
    text = re.sub(r'==+.*?==+', '', text)
    # Collapse multiple consecutive newlines into a clean double newline
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

def build_corpus(topics, output_filename="corpus.txt"):
    """Fetches Wikipedia articles, cleans them, adds <|endoftext|> boundaries, and writes to corpus.txt."""
    wiki = wikipediaapi.Wikipedia(
        user_agent="CustomGPT2Trainer/1.0 (educational_project)",
        language="en"
    )

    total_chars = 0
    successful_topics = 0

    print("🚀 Starting specialized domain scraping...\n")

    with open(output_filename, "w", encoding="utf-8") as f:
        for topic in topics:
            page = wiki.page(topic)
            if page.exists():
                text = clean_text(page.text)
                if len(text) > 200:  # Avoid stub pages
                    # Append article text along with the endoftext boundary token
                    f.write(text + "\n<|endoftext|>\n\n")
                    
                    chars = len(text)
                    total_chars += chars
                    successful_topics += 1
                    print(f"✅ Added '{topic}': {chars:,} characters")
            else:
                print(f"⚠️ Page not found: '{topic}'")

    print("\n" + "="*50)
    print(f"🎉 Processing Complete!")
    print(f"📄 Articles added: {successful_topics}/{len(topics)}")
    print(f"📊 Total corpus size: {total_chars:,} characters (~{total_chars / (1024 * 1024):.2f} MB)")
    print(f"💾 Saved to: {output_filename}")
    print("="*50)

if __name__ == "__main__":
    # Customize this list with topics for your domain!
    target_topics = [
        "Artificial intelligence",
        "Machine learning",
        "Deep learning",
        "Transformer (deep learning architecture)",
        "Natural language processing",
        "Generative pre-trained transformer",
        "Robotics",
        "Autonomous robot",
        "Neural network",
        "Large language model",
        "University of Lagos",
        "Higher education in Nigeria",
        "National Information Technology Development Agency",
        "United Nations Development Programme",
        "Lagos",
        "United Nations",
        "Charter of the United Nations",
        "Nigeria",
        "Nigerians",
        "West Africa",
        "Reinforcement learning",
    "Supervised learning",
    "Unsupervised learning",
    "Convolutional neural network",
    "Recurrent neural network",
    "Attention mechanism (machine learning)",
    "Computer vision",
    "Speech recognition",
    "Data science",
    "Ethics of artificial intelligence",
    "Industrial robot",
    "Internet of things",
    "Embedded system",
    "Microcontroller",
    "Actuator",
    "Sensor",
    "Automation",
    "Unmanned aerial vehicle",
    "Digital manufacturing",
    "Mechatronics",
    "Cloud computing",
    "Application programming interface",
    "Microservices",
    "Database management system",
    "Version control",
    "Open-source software",
    "Cybersecurity",
    "Software engineering",
    "Distributed computing",
    "Data structure",
    "Obafemi Awolowo University",
    "University of Ibadan",
    "Covenant University",
    "Federal University of Technology, Akure",
    "University of Nigeria, Nsukka",
    "Yaba College of Technology",
    "National Universities Commission",
    "Tertiary Education Trust Fund",
    "Academic Staff Union of Universities",
    "Science and technology education",
    "National Centre for Artificial Intelligence and Robotics",
    "Data Science Nigeria",
    "Yaba, Lagos",
    "Digital economy",
    "Information and communications technology in Nigeria",
    "E-governance",
    "Financial technology in Nigeria",
    "Startup ecosystem",
    "African Union",
    "Science and technology in Nigeria",
    "University of Lagos",
    "United Nations Development Programme",
    "Innovation hub",
    "Incubator (business)",
    "Makerspace",
    "Robotics lab",
    "Fab lab",
    "Artificial intelligence",
    "Machine learning",
    "Artificial intelligence of things",
    "Deep learning",
    "Large language model",
    "Natural language processing",
    "Generative artificial intelligence",
    "Computer vision",
    "Robotics",
    "African languages",
    "Low-resource language",
    "Masakhane",
    "Yoruba language",
    "Hausa language",
    "Igbo language",
    "Computational linguistics",
    "Text-to-speech",
    "Speech recognition",
    "Ministry of Communications, Innovation and Digital Economy",
    "National Information Technology Development Agency",
    "Federal Ministry of Education (Nigeria)",
    "Higher education in Nigeria",
    "Tech ecosystem in Nigeria",
    "Economy of Lagos"
    ]

    build_corpus(target_topics)
