# 🚀 Western Ghats BERT - Quick Start Guide

## Installation & Setup

1. **Extract the package**:
   ```bash
   unzip Western_Ghats_BERT_Package.zip
   cd Western_Ghats_BERT_Package
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Test the model**:
   ```bash
   python examples/basic_usage_example.py
   ```

## Basic Usage

```python
from transformers import BertForMaskedLM, BertTokenizerFast

# Load the model
model = BertForMaskedLM.from_pretrained('./western_ghats_bert_improved')
tokenizer = BertTokenizerFast.from_pretrained('./western_ghats_bert_improved')

# Use for masked language modeling
text = "Western Ghats contains many [MASK] species"
inputs = tokenizer(text, return_tensors='pt')
outputs = model(**inputs)
```

## What's Included

- **Model Files**: Complete trained BERT model (~438 MB)
- **Documentation**: Comprehensive README and usage guide
- **Examples**: Ready-to-run Python examples
- **Evaluation**: Model performance metrics and benchmarks
- **Data Samples**: Keywords and metadata used in training

## Next Steps

1. **Read the full README.md** for detailed documentation
2. **Run examples/** to see the model in action
3. **Check evaluation/** for performance metrics
4. **Adapt for your use case** using the provided templates

## Support

- **Documentation**: README.md (comprehensive guide)
- **Examples**: examples/ directory (working code)
- **Citation**: CITATION.bib (for academic use)
- **License**: LICENSE (usage terms)

---
**Created**: 2025-08-18 00:59:41
**Model Version**: Western Ghats BERT v1.0
**Package Size**: ~450 MB (including model weights)
