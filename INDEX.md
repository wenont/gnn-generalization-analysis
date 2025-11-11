# 📚 Documentation Index

Welcome to the GNN Research Project documentation! This index will help you find the information you need quickly.

---

## 🎯 Start Here

**Never used this repository before?**
→ Start with [DOCUMENTATION_SUMMARY.md](DOCUMENTATION_SUMMARY.md)

**Want to get running quickly?**
→ See [QUICK_START.md](QUICK_START.md)

**Need detailed information?**
→ Check [DOCUMENTATION.md](DOCUMENTATION.md)

**Looking for function details?**
→ Browse [CODE_REFERENCE.md](CODE_REFERENCE.md)

---

## 📖 Documentation Files

| File | Purpose | Best For |
|------|---------|----------|
| **[DOCUMENTATION_SUMMARY.md](DOCUMENTATION_SUMMARY.md)** | High-level overview of all docs | First-time users, getting oriented |
| **[QUICK_START.md](QUICK_START.md)** | Fast-track setup and common tasks | Getting started quickly |
| **[DOCUMENTATION.md](DOCUMENTATION.md)** | Comprehensive guide (25+ pages) | In-depth understanding |
| **[CODE_REFERENCE.md](CODE_REFERENCE.md)** | Complete API reference | Developers, advanced users |
| **[README.md](README.md)** | Original project README | Project overview, current tasks |

---

## 🔍 Find Information By Topic

### Installation & Setup
- **Quick setup**: [QUICK_START.md](QUICK_START.md#setup-one-time)
- **Detailed installation**: [DOCUMENTATION.md](DOCUMENTATION.md#installation)
- **Dependencies**: [DOCUMENTATION.md](DOCUMENTATION.md#installation) + [environment.yml](environment.yml)
- **Troubleshooting setup**: [DOCUMENTATION.md](DOCUMENTATION.md#troubleshooting)

### Research Context
- **Research purpose**: [DOCUMENTATION_SUMMARY.md](DOCUMENTATION_SUMMARY.md#research-purpose)
- **Key findings**: [DOCUMENTATION.md](DOCUMENTATION.md#key-findings)
- **Graph parameters**: [DOCUMENTATION_SUMMARY.md](DOCUMENTATION_SUMMARY.md#graph-parameters-computed)
- **Background paper**: [paper/thesis.tex](paper/thesis.tex) + [paper/chapters/introduction.tex](paper/chapters/introduction.tex)

### Using the Code

#### Basic Usage
- **Common tasks**: [QUICK_START.md](QUICK_START.md#common-tasks)
- **Usage guide**: [DOCUMENTATION.md](DOCUMENTATION.md#usage-guide)
- **Typical workflow**: [DOCUMENTATION_SUMMARY.md](DOCUMENTATION_SUMMARY.md#typical-workflow)

#### Computing Graph Parameters
- **Quick command**: [QUICK_START.md](QUICK_START.md#1-compute-graph-parameters)
- **Detailed guide**: [DOCUMENTATION.md](DOCUMENTATION.md#3-computing-graph-parameters)
- **Function reference**: [CODE_REFERENCE.md](CODE_REFERENCE.md#graph-parameters)

#### Training Models
- **Quick training**: [QUICK_START.md](QUICK_START.md#3-train-a-single-model)
- **Detailed guide**: [DOCUMENTATION.md](DOCUMENTATION.md#5-training-models)
- **Function reference**: [CODE_REFERENCE.md](CODE_REFERENCE.md#srctrain.py---training--evaluation)
- **Model architectures**: [CODE_REFERENCE.md](CODE_REFERENCE.md#srcnetpy---neural-network-models)

#### Hyperparameter Optimization
- **Quick W&B setup**: [QUICK_START.md](QUICK_START.md#2-run-hyperparameter-optimization)
- **Detailed guide**: [DOCUMENTATION.md](DOCUMENTATION.md#4-hyperparameter-optimization-with-wb)
- **Configuration**: [CODE_REFERENCE.md](CODE_REFERENCE.md#wb-integration)

#### Analysis & Results
- **Correlation analysis**: [QUICK_START.md](QUICK_START.md#6-correlation-analysis)
- **Generalization error**: [DOCUMENTATION.md](DOCUMENTATION.md#6-computing-generalization-error)
- **Result interpretation**: [DOCUMENTATION.md](DOCUMENTATION.md#results-interpretation)

### Code Structure & API

#### Module Overview
- **Project structure**: [DOCUMENTATION.md](DOCUMENTATION.md#code-structure)
- **Module details**: [CODE_REFERENCE.md](CODE_REFERENCE.md#module-overview)
- **File locations**: [QUICK_START.md](QUICK_START.md#file-locations)

#### Specific Modules
- **net.py (models)**: [CODE_REFERENCE.md](CODE_REFERENCE.md#srcnetpy---neural-network-models)
- **train.py (training)**: [CODE_REFERENCE.md](CODE_REFERENCE.md#srctrain.py---training--evaluation)
- **utils.py (utilities)**: [CODE_REFERENCE.md](CODE_REFERENCE.md#srcutilspy---utility-functions)
- **main.py (analysis)**: [CODE_REFERENCE.md](CODE_REFERENCE.md#srcmainpy---analysis-pipeline)
- **parameter.py (metrics)**: [CODE_REFERENCE.md](CODE_REFERENCE.md#srcparameterpy---graph-metrics-alternative-interface)

#### Functions & Classes
- **TrainParams**: [CODE_REFERENCE.md](CODE_REFERENCE.md#training-configuration)
- **load_dataset()**: [CODE_REFERENCE.md](CODE_REFERENCE.md#dataset-loading)
- **train_procedure()**: [CODE_REFERENCE.md](CODE_REFERENCE.md#main-training-function)
- **get_correlation()**: [CODE_REFERENCE.md](CODE_REFERENCE.md#correlation-analysis)
- **All functions**: [CODE_REFERENCE.md](CODE_REFERENCE.md) (complete reference)

### Datasets
- **Dataset info**: [DOCUMENTATION.md](DOCUMENTATION.md#datasets)
- **Dataset categories**: [DOCUMENTATION_SUMMARY.md](DOCUMENTATION_SUMMARY.md#graph-parameters-computed)
- **TUDataset link**: https://chrsmrrs.github.io/datasets/

### Configuration
- **Model hyperparameters**: [DOCUMENTATION.md](DOCUMENTATION.md#configuration)
- **TrainParams**: [CODE_REFERENCE.md](CODE_REFERENCE.md#training-configuration)
- **W&B sweeps**: [CODE_REFERENCE.md](CODE_REFERENCE.md#wb-integration)

### Troubleshooting
- **Quick fixes**: [DOCUMENTATION_SUMMARY.md](DOCUMENTATION_SUMMARY.md#troubleshooting-quick-reference)
- **Common issues**: [DOCUMENTATION.md](DOCUMENTATION.md#troubleshooting)
- **Error handling**: [CODE_REFERENCE.md](CODE_REFERENCE.md#error-handling)

---

## 🎓 Documentation By User Type

### 👨‍🎓 Student/Researcher (First Time)
**Goal**: Understand the research and run experiments

1. Read [DOCUMENTATION_SUMMARY.md](DOCUMENTATION_SUMMARY.md) (overview)
2. Review [DOCUMENTATION.md](DOCUMENTATION.md#research-purpose) (research context)
3. Follow [QUICK_START.md](QUICK_START.md) (setup)
4. Try [DOCUMENTATION.md](DOCUMENTATION.md#experiment-workflow) (run experiment)
5. Refer to [DOCUMENTATION.md](DOCUMENTATION.md#troubleshooting) (if issues)

### 💻 Developer
**Goal**: Understand and modify the code

1. Skim [DOCUMENTATION_SUMMARY.md](DOCUMENTATION_SUMMARY.md) (overview)
2. Study [DOCUMENTATION.md](DOCUMENTATION.md#code-structure) (architecture)
3. Review [CODE_REFERENCE.md](CODE_REFERENCE.md) (API details)
4. Check inline code comments in `src/` files
5. Use [CODE_REFERENCE.md](CODE_REFERENCE.md#common-patterns) (examples)

### 🔬 Research Supervisor/Reviewer
**Goal**: Understand the methodology and results

1. Read [DOCUMENTATION_SUMMARY.md](DOCUMENTATION_SUMMARY.md#research-purpose) (purpose)
2. Check [paper/](paper/) LaTeX files (thesis content)
3. Review [DOCUMENTATION.md](DOCUMENTATION.md#experiment-workflow) (methodology)
4. See [DOCUMENTATION.md](DOCUMENTATION.md#results-interpretation) (results)
5. Browse [results/](results/) (actual outputs)

### 🚀 Quick Task User
**Goal**: Just need to run something specific

1. Jump to [QUICK_START.md](QUICK_START.md#common-tasks)
2. Find your task
3. Copy-paste command
4. Refer to [DOCUMENTATION.md](DOCUMENTATION.md#troubleshooting) if needed

---

## 📊 Information Depth

```
QUICK_START.md
    ↓ (need more details)
DOCUMENTATION_SUMMARY.md
    ↓ (need comprehensive guide)
DOCUMENTATION.md
    ↓ (need API reference)
CODE_REFERENCE.md
    ↓ (need implementation)
src/*.py files
```

---

## 🔑 Key Sections Quick Links

### Setup
- [Installation](DOCUMENTATION.md#installation)
- [Environment setup](QUICK_START.md#setup-one-time)
- [Dependencies](environment.yml)

### Usage
- [Common tasks](QUICK_START.md#common-tasks)
- [Usage guide](DOCUMENTATION.md#usage-guide)
- [Workflow](DOCUMENTATION.md#experiment-workflow)

### Code
- [Module overview](CODE_REFERENCE.md#module-overview)
- [Function reference](CODE_REFERENCE.md)
- [Common patterns](CODE_REFERENCE.md#common-patterns)

### Results
- [Parameters](DOCUMENTATION.md#results-interpretation)
- [Correlation](DOCUMENTATION.md#7-correlation-analysis)
- [Interpretation](DOCUMENTATION.md#generalization-error-calculation)

### Help
- [Troubleshooting](DOCUMENTATION.md#troubleshooting)
- [Quick fixes](DOCUMENTATION_SUMMARY.md#troubleshooting-quick-reference)
- [Error handling](CODE_REFERENCE.md#error-handling)

---

## 🎯 Quick Decision Tree

**Q: What do you want to do?**

```
Want to understand the project?
    → DOCUMENTATION_SUMMARY.md

Want to run experiments quickly?
    → QUICK_START.md

Want comprehensive information?
    → DOCUMENTATION.md

Want function details?
    → CODE_REFERENCE.md

Want to see research context?
    → paper/thesis.tex

Having problems?
    → DOCUMENTATION.md#troubleshooting

Want to modify code?
    → CODE_REFERENCE.md + src/ files
```

---

## 📂 File Organization

```
Documentation/
├── INDEX.md (this file)           ← You are here
├── DOCUMENTATION_SUMMARY.md       ← Overview of everything
├── QUICK_START.md                 ← Fast-track guide
├── DOCUMENTATION.md               ← Comprehensive guide
├── CODE_REFERENCE.md              ← API reference
└── README.md                      ← Original README

Source Code/
├── src/*.py                       ← Implementation
└── (see DOCUMENTATION.md for details)

Research/
├── paper/                         ← Thesis LaTeX
└── results/                       ← Experimental outputs

Data/
├── data/                          ← Datasets and seeds
└── (see DOCUMENTATION.md for details)
```

---

## 🔄 Update Log

- **2025-02-13**: Initial comprehensive documentation
  - Created DOCUMENTATION.md
  - Created QUICK_START.md
  - Created CODE_REFERENCE.md
  - Created DOCUMENTATION_SUMMARY.md
  - Created INDEX.md

---

## 💡 Tips

1. **Bookmark this page** for quick reference
2. **Use Ctrl+F** to search within documents
3. **Start with QUICK_START** if you're in a hurry
4. **Read DOCUMENTATION** for full understanding
5. **Refer to CODE_REFERENCE** when coding

---

## 📞 Need Help?

Can't find what you need? Try:

1. **Search** in the documentation files (Ctrl+F)
2. **Check** the [troubleshooting section](DOCUMENTATION.md#troubleshooting)
3. **Review** [common issues](DOCUMENTATION_SUMMARY.md#troubleshooting-quick-reference)
4. **Contact** supervisor or advisor
5. **Open** an issue on GitHub (if applicable)

---

## ✨ Documentation Features

✅ **4 levels of detail**: Summary → Quick → Comprehensive → Reference  
✅ **Cross-referenced**: Easy navigation between docs  
✅ **Searchable**: Use Ctrl+F to find topics  
✅ **Example-rich**: Code samples throughout  
✅ **User-focused**: Different paths for different users  
✅ **Complete**: Covers all aspects of the project  
✅ **Up-to-date**: Version 1.0.0 (2025-02-13)  

---

**Happy coding! 🚀**

*Last updated: 2025-02-13*
