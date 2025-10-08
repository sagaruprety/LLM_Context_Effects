# LLM Context Effects: Investigating Order Effects in Similarity Judgements

This repository contains the implementation and analysis for the research paper "Investigating Context Effects in Similarity Judgements in Large Language Models" ([arXiv:2408.10711](https://arxiv.org/pdf/2408.10711)). The study investigates whether Large Language Models (LLMs) exhibit human-like order effects when making similarity judgments between country pairs.

## 📋 Overview

This research replicates a classic study by Tversky and Gati (1978) that demonstrated asymmetric similarity judgments in humans. The key question is: **Do LLMs show order effects when judging similarity between countries, and how do these effects compare to human behavior?**

### Research Question
How well do LLMs align with humans in context-sensitive similarity judgements?

### Key Findings
- Only **Llama3 8B** and **GPT-4** models showed statistically significant order effects similar to humans
- Order effects were highly dependent on temperature settings
- Most LLMs did not exhibit human-like asymmetric similarity judgments
- Results suggest LLMs can be tuned to align with or avoid human-like biases depending on application needs

## 🏗️ Repository Structure

```
LLM_Context_Effects/
├── results/                          # Experimental results and data
│   ├── results_human.csv            # Human baseline data from Tversky & Gati study
│   ├── human_llm_df_ssd_best_models.csv  # Combined human-LLM comparison data
│   └── [various result files]       # Results from different experimental conditions
├── plots/                           # Visualization outputs
│   ├── all_results_violin_plot.pdf
│   └── single_prompt_all_results_0_temp_violin_plot.pdf
├── similarity_effect_*.ipynb        # Jupyter notebooks for different experimental conditions
├── similarity_effect_*.py           # Python scripts for running experiments
├── analyse_*.ipynb                  # Data analysis notebooks
└── create_prompts.ipynb            # Prompt generation utilities
```

## 🚀 Getting Started

### Prerequisites

1. **Python Environment**: Python 3.8+ with conda
2. **API Keys**: 
   - OpenAI API key (for GPT models)
   - Hugging Face API token (for open-source models)
3. **Ollama** (for local model inference): Install from [ollama.ai](https://ollama.ai)

### Installation

#### Option 1: Automated Setup (Recommended)

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd LLM_Context_Effects
   ```

2. **Run the automated setup**:
   ```bash
   python setup.py
   ```
   This will:
   - Create a conda environment named `LLM_Context_Effects`
   - Install all dependencies in the correct environment
   - Set up configuration files
   - Verify Ollama installation

3. **Activate the environment**:
   ```bash
   conda activate LLM_Context_Effects
   ```

4. **Verify installation**:
   ```bash
   python test_installation.py
   ```

#### Option 2: Manual Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd LLM_Context_Effects
   ```

2. **Create and activate conda environment**:
   ```bash
   conda create -n LLM_Context_Effects python=3.9
   conda activate LLM_Context_Effects
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up API keys**:
   ```bash
   cp .env.template .env
   # Edit .env with your actual API keys
   ```

5. **Install and set up Ollama** (for local models):
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Pull required models
   ollama pull llama3.2:3b-instruct-fp16
   ollama pull llama3.1:8b-instruct-fp16
   ollama pull llama3.1:70b-instruct-fp16
   ```

## 🧪 Experimental Design

### Country Pairs
The study uses 21 country pairs from the original Tversky & Gati study:
- U.S.A.-Mexico, U.S.S.R.-Poland, China-Albania
- U.S.A.-Israel, Japan-Philippines, U.S.A.-Canada
- And 15 more pairs...

### Experimental Conditions

1. **Prompt Styles**:
   - **SSA**: "How similar are A and B?"
   - **SST**: "Is A similar to B?"
   - **SSD**: "Assess the degree to which A is similar to B"

2. **Model Types**:
   - **OpenAI**: GPT-4, GPT-3.5-turbo, GPT-4o
   - **Open Source**: Llama2/3 variants, Mistral-7B

3. **Temperature Settings**: 0.001, 0.5, 1.0, 1.5

4. **Order Effects**: Each country pair tested in both orders (A-B vs B-A)

### Key Metrics
- **Similarity Score**: 0-20 scale (0 = no similarity, 20 = complete similarity)
- **Order Difference**: `sim_score_1 - sim_score_2` (difference between orders)
- **Statistical Significance**: p-values for order effect detection

## 🔬 Running Experiments

### Single Prompt Experiments
```bash
python similarity_effect_single_prompt.py
```

### Sampling Experiments (Multiple Trials)
```bash
python similarity_effect_single_prompt.py
```

### Jupyter Notebooks
For interactive analysis and different experimental conditions:
```bash
jupyter notebook similarity_effect_cot_prompt.ipynb
jupyter notebook similarity_effect_dual_prompt.ipynb
```

## 📊 Data Analysis

### Key Analysis Notebooks
- `analyse_data.ipynb`: Main data analysis and visualization
- `analyse_data_sampling_exp.ipynb`: Sampling experiment analysis
- `analyse_reasoning.ipynb`: Chain-of-thought reasoning analysis

### Example Analysis
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('results/human_llm_df_ssd_best_models.csv')

# Check for order effects
order_effects = df.groupby(['model_name', 'temperature'])['sim_diff'].apply(
    lambda x: (x != 0).sum()
).reset_index(name='non_zero_differences')

print(order_effects)
```

## 📈 Results Interpretation

### Human Baseline
- Humans show consistent order effects across country pairs
- Average order difference varies by country pair characteristics

### LLM Behavior
- **Aligned Models**: Llama3 8B, GPT-4 (at specific temperatures)
- **Non-aligned Models**: Most other models show minimal order effects
- **Temperature Sensitivity**: Order effects disappear at higher temperatures

### Statistical Analysis
- Wilcoxon signed-rank tests for order effect significance
- Effect size calculations for practical significance
- Comparison with human baseline data

## 🎯 Key Findings

1. **Limited Alignment**: Only 2 out of 8 tested models showed human-like order effects
2. **Temperature Dependency**: Order effects are highly sensitive to temperature settings
3. **Prompt Sensitivity**: Different prompt styles can influence order effect magnitude
4. **Practical Implications**: 
   - **Consistency needed**: E-commerce, search applications should avoid order effects
   - **Human-like behavior**: Dating, mental health applications might benefit from order effects

## 🔧 Configuration

### Model Configuration
Models are configured in the respective Python files:
```python
model_name_type_mapping = {
    'gpt-4': 'openai',
    'llama3_8B': 'open-source',
    # ... more models
}
```

### Experimental Parameters
```python
temperatures = [0.001, 0.5, 1.0, 1.5]
similarity_effect_country_pairs = [
    ('U.S.A.', 'Mexico'),
    # ... 20 more pairs
]
```

## 📚 Citation

If you use this code or data, please cite:

```bibtex
@article{uprety2024investigating,
  title={Investigating Context Effects in Similarity Judgements in Large Language Models},
  author={Uprety, Sagar and Jaiswal, Amit Kumar and Liu, Haiming and Song, Dawei},
  journal={arXiv preprint arXiv:2408.10711},
  year={2024}
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 References

- [Original Paper](https://arxiv.org/pdf/2408.10711)
- [Tversky & Gati (1978) - Original Human Study](https://api.semanticscholar.org/CorpusID:9173202)
- [LangChain Documentation](https://python.langchain.com/)
- [Ollama Documentation](https://ollama.ai/docs)

## 🆘 Troubleshooting

### Common Issues

1. **Dependencies not installed in conda environment**:
   ```bash
   # If setup.py didn't work correctly, run:
   python fix_dependencies.py
   ```

2. **API Key Errors**: 
   ```bash
   # Set environment variables
   export OPENAI_API_KEY="your-key-here"
   # Or create .env file
   cp .env.template .env
   # Edit .env with your keys
   ```

3. **Ollama Connection**: 
   ```bash
   # Start Ollama server
   ollama serve
   # In another terminal, test connection
   ollama list
   ```

4. **Memory Issues**: Use smaller models or reduce batch sizes for local inference

5. **CUDA Issues**: Set `CUDA_VISIBLE_DEVICES` or use CPU-only mode

6. **Module not found errors**:
   ```bash
   # Make sure you're in the right conda environment
   conda activate LLM_Context_Effects
   # Verify installation
   python test_installation.py
   ```

### Quick Fixes

- **Reinstall dependencies**: `python fix_dependencies.py`
- **Test installation**: `python test_installation.py`
- **Check conda environment**: `conda info --envs`

### Getting Help

- Check the [Issues](https://github.com/your-repo/issues) page
- Review the analysis notebooks for examples
- Consult the original paper for methodological details

---

**Note**: This research investigates psychological biases in AI systems. The findings have important implications for AI alignment and the design of LLM-based applications.
