# 🌉 InterOrdra

*Semantic gap detection framework - Makes the invisible structure of misunderstanding visible*

**🌐 Currently available in Spanish. English version coming soon.**

**🚀 Live Demo:** https://interordra.streamlit.app

---

## 🎯 What It Does

InterOrdra analyzes any pair of texts and reveals exactly where and why they don't understand each other - detecting gaps in vocabulary, orphaned concepts, and topological misalignment in semantic space.

InterOrdra reveals **why** by analyzing the semantic topology between any two texts.

- 📚 **Technical documentation vs user questions** - Find what's missing in your docs
- 🤖 **AI prompts vs expected responses** - Optimize prompt engineering  
- 🔬 **Scientific papers vs popular articles** - Measure accessibility
- 💼 **Expert explanations vs public understanding** - Bridge communication gaps

## How It Works

1. **Semantic Embeddings** - Converts texts into high-dimensional vectors capturing meaning
2. **Topological Analysis** - Maps conceptual clusters and identifies orphaned ideas
3. **Gap Detection** - Calculates similarity matrices to find misalignments
4. **3D Visualization** - Interactive visualization of semantic topology
5. **Actionable Recommendations** - Concrete suggestions to close the gaps

## Tech Stack

- Python 3.11
- Sentence Transformers (semantic embeddings)
- Scikit-learn (clustering, similarity)
- Streamlit (web interface)
- Plotly (3D visualization)

## Installation
```bash
git clone https://github.com/rosibis-piedra/interordra.git
cd interordra
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run frontend/app.py
```

## Project Structure
```
interordra/
├── backend/
│   ├── embeddings.py      # Semantic vector generation
│   ├── clustering.py      # Concept clustering
│   ├── gap_detector.py    # Gap identification
│   ├── analyzer.py        # Main pipeline
│   └── simple_splitter.py # Text processing
├── frontend/
│   └── app.py            # Streamlit interface
└── requirements.txt
```

## Roadmap

- [ ] English UI translation
- [ ] PDF export of analysis results
- [ ] File upload support (.txt, .docx, .pdf)
- [ ] Multi-text comparison (3+ texts)
- [ ] API for developers
- [ ] Temporal pattern analysis (Phase 2)

## About

Created by [Rosibis](https://github.com/rosibis-piedra) - Semantic bridge architect

Part of a broader vision: building a **Resonance Spectrometer** to detect coordinated pattern transmission across all communication bands.

## License

MIT

---

*"Language is not a thing humans have and other systems lack. It is a narrow band within a vast spectrum of coordinated pattern transmission."*
