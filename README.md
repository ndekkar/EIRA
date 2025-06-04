# EIRA: Energy Infrastructure Risk Assessment Tool

**EIRA** is a Python-based tool designed to facilitate multi-hazard risk assessments in the **energy sector**. It streamlines geospatial data processing and enables users to analyze exposure, vulnerability, and risk using a modular and scalable architecture.

EIRI can works and analizes the impact of several hazards, population and economic exposure maily related to the energy sector. EIRA can provide thier own data for part of all the risk analysis. The hazards than can be analized with EIRA are listed in below. 

The current version EIRA v1.0 is not able to run probabilistic risk analysis, however, the structure of the programming was designed to introduce those new capabilities in future version of the tool (EIRA v2.0).  

EIRA v1.0 provides several classes, methods and data for exposure, hazard and vuleranbility risk analysis.  

## 🔍 Overview

EIRA is composed of several key modules:
- **Hazard**
- **Vulnerability**
- **Exposure**
- **Risk Assessment** (at various levels of detail)

This tool is particularly useful for handling large datasets, especially geospatial files related to exposure and hazards. It supports automated workflows and integrates seamlessly with remote repositories for downloading essential datasets into Jupyter Notebooks.

## 🌍 Purpose

The primary goal of EIRA is to compute risk in the energy sector using different levels of information detail. EIRA v1.0 It provides a framework for users to combine exposure, hazard and vulnerability and to calculate/estimate several risk indicators and metric of the risk in the energy sector.

The **first level of risk assessment** relies on the spatial overlap of hazard and exposure layers. Based on this, EIRA calculates how many energy assets are exposed to various intensities of different natural hazards, including:
- Earthquakes
- Floods
- Landslides
- Tropical Cyclones (Windstorms)
- Storm surges
- High temperatures
- Droughts
- Tsnami *
- And more...

This functionality supports preliminary risk mapping and decision-making for energy infrastructure resilience and planning.

## 🔧 Features

- Modular design for hazard, exposure, vulnerability, and risk modules
- Efficient handling of large-scale geospatial datasets
- Remote connection to online repositories for dynamic data access
- Jupyter Notebook interface for interactive analysis
- Support for multi-hazard risk assessment workflows

## 🚀 Getting Started

To set up and run EIRA on your local machine, follow these steps:

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/eira.git
cd eira
````

### 2. Create a Conda Environment
Create a new Conda environment named EIRAtool_env with Python 3.10:
```bash
conda create --name EIRAtool_env python=3.10
```

### 3. Activate the Environment
Activate the newly created Conda environment:
```bash
conda activate EIRAtool_env
```

### 4. Install Dependencies
Install all required Python libraries using the requirements.txt file:
```bash
pip install -r requirementsEira_v1.txt
```
