# agent-cv

Agentic system to automatically train portable computer vision models using YOLOv9, Grounding DINO, and vision-language models for automated labeling and iterative training.

## Overview

This system combines AI agents with computer vision tools to automatically:
- Generate high-quality labels for unlabeled datasets using Grounding DINO and VLMs
- Train YOLOv9 models with iterative refinement
- Generate deployment code for portable model conversion

## Architecture

**Core Components:**
- **DataLoader**: Data ingestion, preprocessing, validation  
- **AutoLabeler**: Grounding DINO + VLM labeling pipeline

**Agents:**
- **OrchestratorAgent**: Workflow coordination and decision-making
- **TrainingAgent**: YOLOv9 training orchestration and optimization
- **EvaluationAgent**: Model assessment and validation
- **DeploymentAgent**: Model conversion and deployment code generation

## Installation

```bash
pip install -e .
```

## Requirements

- Python 3.12+
- PyTorch
- Anthropic API key (for VLM labeling)

## Usage

```bash
python main.py
```

## Project Structure

```
agent_cv/
├── agents/          # Agent implementations
├── models/          # Data models for communication
├── tools/           # Utility functions and data processing
├── config.py        # Configuration management
└── lmm.py          # Large multimodal model integration
```

## Development Plan

1. **Set up project skeleton:**
   - Organize directories and create initial module files
   - Include configuration files such as `pyproject.toml`, `.env.example`, and `README.md`

2. **Build interface functions:**
   - Implement core functions in `agent_cv/tools/` for data loading, preprocessing, training, and evaluation
   - Implement data models in `agent_cv/models/` for communication and parsing
   - Implement tool search with encoding for agent tool retrieval

3. **Implement agent components:**
   - Develop agent classes in `agent_cv/agents/` to orchestrate workflows and manage state
   - Create OrchestratorAgent for workflow coordination
   - Build TrainingAgent, EvaluationAgent, and DeploymentAgent

4. **Integrate automatic labeling:**
   - Add support for automatic data labeling using Grounding DINO for object detection
   - Implement VLM integration for semantic labeling

5. **Iterate and refine:**
   - Test end-to-end workflows
   - Improve agent logic and expand toolset as needed
