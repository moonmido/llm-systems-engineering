# 🤖 LLM Systems Engineering

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)

**A comprehensive, engineering-focused repository covering Large Language Models (LLMs) — from Transformer internals to production-grade AI system design.**

*Practical implementations · Hands-on experiments · Modern techniques*

</div>

---

## 📖 About

This repository is a structured, chapter-by-chapter deep dive into the engineering and systems side of Large Language Models. It goes beyond surface-level tutorials to explore the mechanics, architecture, and production considerations that make LLMs work at scale.

Whether you're transitioning into AI engineering, preparing for ML system design interviews, or building real-world LLM applications, this repo provides the practical foundation you need — through working Jupyter notebooks, clean Python code, and carefully crafted experiments.

**Topics covered include:**

- Transformer architecture & attention mechanisms
- Tokenization & embeddings
- Prompt engineering strategies
- Retrieval-Augmented Generation (RAG)
- Fine-tuning & parameter-efficient adaptation (LoRA, QLoRA, PEFT)
- Vector databases & semantic search
- LLM evaluation & benchmarking
- Inference optimization & quantization
- Modern AI system design patterns

---

## 🗂️ Repository Structure

```
llm-systems-engineering/
├── Chapter-1-/         # Foundations: Transformers & LLM Architecture
├── Chapter-2-/         # Embeddings, Tokenization & Semantic Search
├── Chapter-3-/         # Prompt Engineering & RAG
├── Chapter-4-/         # Fine-Tuning & Adaptation
├── Chapter-5-/         # Evaluation, Inference Optimization & System Design
├── Chapter-6-/         # ReAct-based AI agent (WorldCupGPT) using LangChain and NVIDIA Chat API.
├── LICENSE
└── README.md
```

---

## 📚 Chapters Overview

### 📌 Chapter 1 — Foundations: Transformers & LLM Architecture

> *Understanding the engine before driving the car.*

This chapter lays the theoretical and practical groundwork for everything that follows. You'll implement and dissect the core components of the Transformer architecture from scratch.

**Key topics:**
- The Transformer architecture: encoders, decoders, encoder-decoders
- Self-attention and multi-head attention mechanisms
- Positional encodings (absolute vs. rotary)
- Autoregressive text generation: logits, sampling strategies (greedy, top-k, top-p, temperature)
- KV-cache and why it matters for inference speed
- Overview of prominent model families: GPT, BERT, T5, LLaMA

---

### 📌 Chapter 2 — Embeddings, Tokenization & Semantic Search

> *How machines read: turning text into numbers that carry meaning.*

A focused exploration of how text is represented numerically, and how those representations enable powerful downstream tasks like semantic similarity, clustering, and retrieval.

**Key topics:**
- Byte-Pair Encoding (BPE), WordPiece, and SentencePiece tokenization
- Word embeddings vs. contextual embeddings
- Sentence and document-level embeddings
- Cosine similarity and vector space geometry
- Building a semantic search engine from scratch
- Introduction to vector databases: FAISS, ChromaDB, Pinecone, Weaviate
- Embedding model benchmarking (MTEB)

---

### 📌 Chapter 3 — Prompt Engineering & Retrieval-Augmented Generation (RAG)

> *Getting the most out of a model — without changing its weights.*

This chapter covers the art and science of prompt design, then scales up to full RAG pipelines that ground LLM outputs in external, up-to-date knowledge.

**Key topics:**
- Zero-shot, few-shot, and chain-of-thought (CoT) prompting
- ReAct, self-consistency, and tree-of-thought techniques
- Prompt templates and system message design
- RAG architecture: retrieval, re-ranking, augmentation, generation
- Chunking strategies for document indexing
- Hybrid search: dense + sparse (BM25) retrieval
- Advanced RAG: query expansion, HyDE, multi-query retrieval
- Guardrails and hallucination mitigation

---

### 📌 Chapter 4 — Fine-Tuning & Parameter-Efficient Adaptation

> *Teaching an old model new tricks — efficiently.*

Full fine-tuning a large model is rarely practical. This chapter covers modern adaptation techniques that deliver strong performance with a fraction of the compute.

**Key topics:**
- When to fine-tune vs. prompt engineering
- Supervised Fine-Tuning (SFT) on instruction datasets
- Parameter-Efficient Fine-Tuning (PEFT): LoRA, QLoRA, Adapters, Prefix Tuning
- Dataset preparation: formatting, cleaning, and augmentation
- Training configuration: learning rate schedules, gradient checkpointing, mixed precision
- RLHF overview: reward models and PPO
- Direct Preference Optimization (DPO) as an RLHF alternative
- Model merging techniques (SLERP, TIES, DARE)

---

### 📌 Chapter 5 — Evaluation, Inference Optimization & System Design

> *How do you know it works? How do you make it fast? How do you ship it?*

The final chapter bridges the gap between a working prototype and a production-ready LLM system, covering evaluation rigor and inference-time engineering.

**Key topics:**
- LLM evaluation metrics: perplexity, BLEU, ROUGE, BERTScore
- LLM-as-a-judge and MT-Bench style evaluation
- RAG evaluation: RAGAS framework, faithfulness, relevance, context recall
- Quantization: GPTQ, AWQ, GGUF, bitsandbytes
- Speculative decoding and continuous batching
- Serving engines: vLLM, TGI, Ollama
- Latency profiling: TTFT, TPOT, throughput, and p95 latency
- AI system design patterns: caching, fallbacks, monitoring, and observability

---


### 📌 Chapter 6 — ReAct-based AI agent

> *Teaching an old model new tricks — efficiently.*

How do you build an intelligent agent that can reason, search the web, and act like a domain expert?
This chapter focuses on building a ReAct-based AI agent (WorldCupGPT) using LangChain, NVIDIA LLMs, and external tools to create a real-time football analyst for the FIFA World Cup 2026.

**Key topics:**
- Building a ReAct (Reasoning + Acting) agent architecture
- Integrating ChatNVIDIA (NVIDIA AI Endpoints) as the core LLM
- Designing a domain-specific system prompt (World Cup 2026 expert)
- Adding tool use capability (DuckDuckGo search)
- Implementing real-time web-augmented reasoning
- Structuring agent memory with LangGraph InMemorySaver
- Managing multi-turn conversations with thread-based execution
- Combining LLM reasoning + external search tools
- Designing a football analytics AI assistant (teams, tactics, predictions)
---

### 📌 Chapter 7 — ReAct-based Multi-Agent Startup Factory

> *Building intelligent systems that don’t just think — but build, design, and generate software autonomously.*

This chapter focuses on designing a ReAct-driven multi-agent AI pipeline (StartupForge) that transforms a simple idea into a complete software system using reasoning + tools + structured agent chaining.

Unlike single-agent ReAct systems (like WorldCupGPT), this chapter extends the concept into a multi-stage production pipeline where each agent becomes a specialized worker in a startup factory.

**Key topics:**
- 🧠 1. ReAct as Multi-Agent Architecture Core
Extending ReAct (Reason + Act) beyond a single loop
Turning each stage into an independent reasoning agent
Sequential reasoning pipelines (not just one agent loop)

- ⚙️ 2. Designing a Multi-Agent Pipeline (StartupForge)
Idea Validator Agent (feasibility reasoning)
Market Research Agent (real-world intelligence extraction)
Product Manager Agent (MVP PRD generation)
System Design Agent (microservices + architecture design)
Coding Agent (production-ready code generation)

- 🔗 3. Tool-Based Agent Orchestration
Using @tool abstraction for each agent
Converting LLMs into callable system components
Supervisor vs deterministic pipeline design

- 🧩 4. Structured Inter-Agent Communication
Pydantic-based schemas for every stage
Enforcing strict output contracts between agents
Preventing hallucination propagation across pipeline

- 🔄 5. State-Passing Pipeline Architecture
Passing outputs between agents:
Idea → Validation → Market → PRD → System Design → Code
Eliminating memory loss between steps
Building a traceable execution flow

- 🏗 6. System Design as an Intermediate Representation (IR)
System Design Agent becomes the “source of truth”
Converting PRD → architecture → services → APIs → databases
Preparing deterministic code generation input

- 🚀 7. Code Generation Agent (Production Layer)
Converting system design into real backend implementations
Spring Boot / Microservices architecture generation
Clean Architecture + event-driven system support

- 🔁 8. ReAct vs Pipeline Agent Design
ReAct loop (single agent reasoning + tool usage)
vs
Deterministic multi-agent pipeline (your system)

Key insight:

StartupForge uses ReAct internally, but externally behaves like a deterministic production system

- 🧠 9. NVIDIA LLM Integration for Agent Roles
Using different models per agent role:
Fast reasoning models → validation & research
Strong reasoning models → system design & coding
Optimizing cost vs intelligence per stage

- ⚡ 10. Production Multi-Agent Engineering Patterns
Supervisor orchestration vs pipeline chaining
Tool-based agent composition
Error handling between agent stages
Structured logging and traceability
---



## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- [Jupyter Notebook](https://jupyter.org/install) or [JupyterLab](https://jupyterlab.readthedocs.io/)
- A GPU is recommended for Chapters 4 & 5 (Google Colab works fine)

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/moonmido/llm-systems-engineering.git
cd llm-systems-engineering
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

3. **Install dependencies**

Each chapter folder contains its own `requirements.txt`. Install for a specific chapter:



```bash
pip install torch transformers datasets sentence-transformers \
            faiss-cpu chromadb langchain openai tqdm jupyter
```

4. **Launch Jupyter**

```bash
jupyter notebook
```

---

## 🛠️ Technologies & Tools

| Category | Tools |
|---|---|
| **Core ML** | PyTorch, Hugging Face Transformers, Datasets |
| **Embeddings & Search** | Sentence-Transformers, FAISS, ChromaDB |
| **LLM Frameworks** | LangChain, LlamaIndex |
| **Fine-Tuning** | PEFT, TRL, Unsloth, bitsandbytes |
| **Evaluation** | RAGAS, evaluate, BERTScore |
| **Inference** | vLLM, Ollama, GGUF/llama.cpp |
| **Notebooks** | Jupyter, Google Colab |

---

## 💡 How to Use This Repository

This repo is designed to be worked through **chapter by chapter**, with each chapter building on the previous. That said, each chapter is self-contained enough to be explored independently if you already have background in a particular area.

- **Beginners:** Start at Chapter 1 and progress linearly.
- **Practitioners:** Jump to Chapter 3 (RAG) or Chapter 4 (fine-tuning) based on your current project needs.
- **ML Engineers:** Chapter 5 covers inference optimization and system design patterns most relevant to production deployments.

Each notebook includes:
- Conceptual explanations with diagrams and intuition
- Runnable code cells with inline comments
- Experiments you can tweak and extend
- References to key papers and resources

---

## 📋 Prerequisites & Recommended Background

| Topic | Level Needed |
|---|---|
| Python | Intermediate |
| Linear Algebra & Probability | Basics |
| Neural Networks / Deep Learning | Helpful but not required |
| PyTorch | Beginner |
| Cloud / GPU Access | For Chapters 4–5 |

---

## 🤝 Contributing

Contributions are welcome! If you'd like to improve a notebook, fix a bug, add experiments, or extend a chapter:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-improvement`
3. Commit your changes: `git commit -m "Add experiment on LoRA rank sensitivity"`
4. Push to your fork: `git push origin feature/my-improvement`
5. Open a Pull Request

Please keep notebooks clean and well-commented. Include a brief description of what you added or changed in the PR.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Boutmedjet Abd elmoudjib**
- GitHub: [@moonmido](https://github.com/moonmido)

---

## ⭐ Acknowledgements

This repository draws inspiration from and builds upon the excellent work of the broader LLM community, including:

- [Hugging Face](https://huggingface.co/) for the Transformers ecosystem
- [Andrej Karpathy](https://github.com/karpathy) for foundational educational content
- The authors of key papers: Attention Is All You Need, LoRA, RAG, RLHF, DPO, and more

---

<div align="center">
  <sub>If you find this repository useful, consider giving it a ⭐ — it helps others discover it!</sub>
</div>
