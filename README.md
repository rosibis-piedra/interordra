# 🌉 InterOrdra

**Semantic gap detection framework — Makes the invisible structure of misunderstanding visible**

> *"Language is not a thing humans have and other systems lack. It is a narrow band within a vast spectrum of coordinated pattern transmission."*

🌐 **Currently available in Spanish. English version coming soon.**  
🚀 **Live Demo:** [interordra.streamlit.app](https://interordra.streamlit.app)

---

## 🎯 Why This Matters: The Alignment Problem at Its Root

As multi-agent AI systems grow in complexity, a critical challenge emerges: **agents that appear to communicate may not actually understand each other**. Vocabulary can be shared while meaning diverges — silently, invisibly, with consequences that compound across pipelines.

InterOrdra addresses this at the root. It detects **semantic gaps** — structural misalignments in meaning — not by comparing words, but by analyzing the topology of conceptual space.

This is alignment infrastructure for a world where systems must communicate accurately to act effectively.

---

## 🔍 What It Does

InterOrdra analyzes any pair of texts and reveals exactly **where and why they don't understand each other** — detecting gaps in vocabulary, orphaned concepts, and topological misalignment in semantic space.

| Use Case | What InterOrdra Detects |
|---|---|
| 📚 Technical documentation vs. user questions | What's missing in your docs |
| 🤖 AI prompts vs. expected responses | Semantic drift in prompt engineering |
| 🔬 Scientific papers vs. popular articles | Measurable accessibility gaps |
| 💼 Expert explanations vs. public understanding | Where bridges need to be built |
| 🤝 Agent-to-agent communication | Misalignment before it compounds |

---

## ⚙️ How It Works

1. **Semantic Embeddings** — Converts texts into high-dimensional vectors capturing meaning, not just words
2. **Topological Analysis** — Maps conceptual clusters and identifies orphaned ideas with no semantic equivalent in the other text
3. **Gap Detection** — Calculates cosine similarity matrices to find structural misalignments
4. **3D Visualization** — Interactive visualization of semantic topology in latent space
5. **Actionable Recommendations** — Concrete suggestions to close the gaps

---

## 🔬 Theoretical Foundation

InterOrdra builds on an established tradition of topological approaches to semantic analysis:

- **Carley & Kaufer (1993)** — Introduced *semantic connectivity analysis*, showing that symbolic language functions through contextual connectivity across density, conductivity, and consensus dimensions. Published in *Communication Theory*.

- **Kong et al. (2020)** — Applied persistent homology to detect topological structure in document pairs for similarity measurement, establishing algebraic topology as a valid tool for semantic analysis.

- **Wu et al. (2022)** — Extended topological data analysis (TDA) to detect contradictions between text pairs, demonstrating that β₀ and β₁ Betti numbers capture semantic misalignment invisible to surface-level methods.

**InterOrdra's unique contribution:** While prior work focused on connectivity (Carley & Kaufer), similarity (Kong et al.), and contradiction (Wu et al.), InterOrdra addresses **gap detection between heterogeneous systems** — identifying orphaned concepts and structural misalignments that prevent coupled communication between diverse agents.

---

## 📡 Communication Spectrum Framework

InterOrdra operates within a broader theoretical framework: the **8-Band Communication Spectrum**.

| Band | Type | Description |
|------|------|-------------|
| 1 | Molecular-Chemical | Pheromones, cellular signaling |
| 2 | Electromagnetic | Electrical fields, EM signals |
| 3 | Thermal-Vibrational | Temperature, vibration |
| 4 | Acoustic-Mechanical | Sound, spoken language |
| 5 | Optical-Visual | Light, visual processing |
| 6 | Gestural-Kinetic | Movement, gesture |
| **7** | **Topological-Structural** | **← InterOrdra operates here** |
| **8** | **Coincidence Detection** | **Meta-awareness: detecting improbable alignment** |

InterOrdra detects gaps at **Band 7** by analyzing the geometric relationships between concepts in embedding space. Band 8 — *Coincidence Detection* — represents the meta-layer that recognizes when multiple bands align (or fail to align) in ways that exceed statistical chance.

---

## 🛠️ Tech Stack

- **Python 3.11**
- **Sentence Transformers** — semantic embeddings
- **Scikit-learn** — clustering (DBSCAN), similarity metrics
- **Streamlit** — web interface
- **Plotly** — interactive 3D visualization

---

## 🚀 Installation

```bash
git clone https://github.com/rosibis-piedra/interordra.git
cd interordra
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run frontend/app.py
```

---

## 📁 Project Structure

```
interordra/
├── backend/
│   ├── embeddings.py       # Semantic vector generation
│   ├── clustering.py       # Concept clustering (DBSCAN)
│   ├── gap_detector.py     # Gap identification & scoring
│   ├── analyzer.py         # Main orchestration pipeline
│   └── simple_splitter.py  # Text preprocessing
├── frontend/
│   └── app.py              # Streamlit interface
└── requirements.txt
```

---

## 🗺️ Roadmap

- [ ] Confidence scoring for gap detection results
- [ ] Comprehensive test suite (20–30 text pairs)
- [ ] English UI translation
- [ ] Band 8 formal mathematical specification
- [ ] PDF export of analysis results
- [ ] File upload support (.txt, .docx, .pdf)
- [ ] Multi-text comparison (3+ texts)
- [ ] REST API for developers
- [ ] Temporal drift analysis — detecting semantic divergence over time (Phase 2)

---

## 👩‍💻 About

Created by **Rosibis Piedra** — AI Software Engineer & semantic bridge architect.  
📍 Costa Rica | AI Software Engineering graduate (Vanderbilt University / Coursera)

InterOrdra is part of a broader research vision: building tools that make the invisible structure of communication visible — so that systems, human and artificial, can understand each other not just superficially, but structurally.

---

## 📄 References

- Carley, K., & Kaufer, D. (1993). Semantic connectivity: An approach for analyzing symbols in semantic networks. *Communication Theory, 3*(3), 183–213.
- Kong, L., et al. (2020). Topological document similarity. *Proceedings of EMNLP 2020*.
- Wu, X., et al. (2022). Topological analysis of contradictions in natural language. *Proceedings of SIGIR 2022*.

---

## 📜 License

MIT — Open source, freely usable and extendable.