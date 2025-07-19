# agent-cv

Agentic system for automatically training computer vision models. Combines AI agents with YOLO training capabilities and Jupyter-based code execution for automated dataset analysis and model training.

## Overview

This system provides an automated workflow for computer vision model development:
- **Dataset Analysis**: Multi-format support (YOLO, COCO, Pascal VOC) with comprehensive analysis
- **Model Training**: Automated YOLO training with configurable parameters using MultimediaTechLab YOLO
- **Code Execution**: Jupyter notebook environment for dynamic experimentation
- **Multi-modal AI**: Integration with Anthropic's Claude for vision-language tasks

## Architecture

**Core Components:**
- **DatasetAnalyzer** (`agent_cv/data_processing/`): Multi-format dataset analysis and validation
- **CodeInterpreter** (`agent_cv/execution/`): Jupyter notebook-based code execution environment
- **AnthropicLMM** (`agent_cv/lmm/`): Large multimodal model with image support

**Data Models:**
- **AgentMessage**: Inter-agent communication with multimodal content support
- **DatasetInfo, TrainingConfig, TrainingResult**: Structured data models for ML workflows
- **TaskStatus, TaskResult**: Task management and execution tracking

**Actions System:**
- **Dataset Actions**: Dataset analyzer creation and management
- **Training Actions**: Full YOLO training pipeline with Lightning integration

## Installation

```bash
pip install -e .
```

## Requirements

- Python 3.12+
- PyTorch
- Anthropic API key (for multimodal AI capabilities)

## Usage

```bash
python main.py
```

## Project Structure

```
agent_cv/
├── actions/            # Agent-callable action functions
│   ├── dataset_actions.py    # Dataset analyzer creation
│   └── training_actions.py   # YOLO training pipeline
├── agents/             # AI agent implementations
│   └── training_agent.py     # Training orchestration agent
├── data_processing/    # Dataset analysis and preprocessing
│   └── dataset_analyzer.py   # Multi-format dataset analysis
├── execution/          # Code execution environment
│   └── interpreter.py        # Jupyter notebook interpreter
├── lmm/               # Large multimodal model integration
│   └── lmm.py                # Anthropic Claude integration
├── models/            # Data models and communication
│   ├── dataset.py            # Dataset information models
│   ├── messages.py           # Agent communication models
│   └── training.py           # Training configuration/results
├── prompts/           # Agent prompts and templates
│   ├── orchestrator_prompts.py
│   └── training_prompts.py
└── utils/             # Utility functions
    ├── formatting.py         # Markup normalization
    └── logs.py               # Rich logging utilities
```

## Features

### Dataset Analysis
- **Multi-format Support**: YOLO, COCO, Pascal VOC annotation formats
- **Comprehensive Analysis**: Class distribution, train/val/test splits, metadata extraction
- **Sample Generation**: Random sampling for dataset inspection

### Model Training
- **YOLO Integration**: Full training pipeline using MultimediaTechLab YOLO
- **Lightning Framework**: Robust training with callbacks, logging, and checkpoints
- **Configurable Parameters**: Flexible training configuration with hyperparameter support
- **Metrics Tracking**: Comprehensive training metrics (mAP, precision, recall, losses)

### Code Execution
- **Jupyter Environment**: Full notebook environment with persistent kernel
- **Execution Tracking**: Detailed logging of stdout, stderr, and exit codes
- **Error Handling**: Robust exception handling and traceback capture

### Multimodal AI
- **Vision-Language Models**: Integration with Anthropic's Claude for image analysis
- **Flexible Content**: Support for text, images, and file paths in agent messages
- **Media Processing**: Automatic base64 encoding for image files

## Implementation Status

The codebase currently implements:
- ✅ Core data models and communication framework
- ✅ Multi-format dataset analysis (YOLO, COCO, Pascal VOC)
- ✅ Full YOLO training pipeline with Lightning integration
- ✅ Jupyter-based code execution environment
- ✅ Anthropic LMM integration with multimodal support
- ✅ Rich logging and formatting utilities
- 🚧Training agent framework with prompt templates
- 🚧 End-to-end agent orchestration workflows
- 🚧 Automated labeling pipeline integration
