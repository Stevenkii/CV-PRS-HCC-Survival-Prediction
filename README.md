# CV-PRS-HCC-Survival-Prediction
Official code for "Vascular Architecture Principles in Hepatocellular Carcinoma from Routine H&amp;E: An Interpretable Framework Linking Vascular Organization, Immunity, and Therapy Benefit"

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Status: Under Review](https://img.shields.io/badge/Status-Under_Review-orange)]()

Official implementation of the manuscript: **"Vascular Architecture Principles in Hepatocellular Carcinoma from Routine H&E: An Interpretable Framework Linking Vascular Organization, Immunity, and Therapy Benefit"**

## 🌟 Overview
We provide an interpretable computational pathology framework linking vascular organization, immunity, and therapy benefit in Hepatocellular Carcinoma (HCC). 
- **Upstream:** PaSegNet (Tissue Seg) -> Omni-Seg (Vessel) -> CellViT++ (Cell Spatial Metrics).
- **Core:** SurvFormer architecture for prognostic risk score (**CV-PRS**) modeling.
- **Downstream:** Integration with genomic TCR Repertoire and clinical variables.

## 📂 Code Availability Notice
**[ Notice to Reviewers & Community ]**
To ensure research integrity and adhere to institutional data-sharing policies (HaploX Biotechnology / Shenzhen Third People's Hospital), the full source code, pre-trained network weights, and reproducible configuration files will be progressively released upon the formal acceptance of the manuscript. Currently, we are structuring the core repositories.
