# Supervised-Contrastive-Ordinal-Loss
**Abstract:**
Abdominal Aortic Calcification (AAC) is a known marker of asymptomatic Atherosclerotic Cardiovascular Diseases (ASCVDs). AAC can be observed on Vertebral Fracture Assessment (VFA) scans acquired using Dual-Energy X-ray Absorptiometry (DXA) machines. Thus, the automatic quantification of AAC on VFA DXA scans may be used to screen for CVD risks, allowing early interventions.  In this research, we formulate the quantification of AAC as an ordinal regression problem. We propose a novel Supervised Contrastive Ordinal Loss (SCOL) by incorporating a label-dependent distance metric with existing supervised contrastive loss to leverage the ordinal information inherent in discrete AAC regression labels. Furthermore, we develop a Dual-encoder Contrastive Ordinal Learning (DCOL)  framework that learns the contrastive ordinal representation at global and local levels to improve the feature separability and class diversity in latent space among the AAC genera. We evaluate the performance of the proposed framework using two clinical VFA DXA scan datasets and compare our work with state-of-the-art methods. Furthermore, for predicted AAC scores, we provide a clinical analysis to predict the future risk of a Major Acute Cardiovascular Event (MACE). Our results demonstrate that this learning enhances inter-class separability and strengthens intra-class consistency among the AAC-24 genera, which results in predicting the high-risk AAC classes with high sensitivity and high accuracy.



Code for MICCAI 2023 publication: SCOL: Supervised Contrastive Ordinal Loss for Abdominal Aortic Calcification Scoring on Vertebral Fracture Assessment Scans



![framework](https://github.com/AfsahS/Supervised-Contrastive-Ordinal-Loss-for-Ordinal-Regression/assets/52653609/92338fff-0f16-4ac3-98b8-7f57c647f83d)



![SCOL_loss](https://github.com/AfsahS/Supervised-Contrastive-Ordinal-Loss-for-Ordinal-Regression/assets/52653609/c3647b6b-2907-45c8-a66f-7270234e6dd2)



![table1](https://github.com/AfsahS/Supervised-Contrastive-Ordinal-Loss-for-Ordinal-Regression/assets/52653609/3d34e7c0-ae82-4892-9320-22fbe7079bae)

![table 2](https://github.com/AfsahS/Supervised-Contrastive-Ordinal-Loss-for-Ordinal-Regression/assets/52653609/d8027112-1ac8-4bc1-8161-c4009dcf6ccf)

![table3](https://github.com/AfsahS/Supervised-Contrastive-Ordinal-Loss-for-Ordinal-Regression/assets/52653609/f4aa1256-319f-4626-a41a-4b394ac9e134)



**Confusion matrix for Table 1**
![comp_CM](https://github.com/AfsahS/Supervised-Contrastive-Ordinal-Loss-for-Ordinal-Regression/assets/52653609/1e3b8b59-22b5-4f54-bc7e-71ca24822c5b)

**Confusion matrix for Table 2**
![CM_baseline](https://github.com/AfsahS/Supervised-Contrastive-Ordinal-Loss-for-Ordinal-Regression/assets/52653609/494873f9-ec87-4205-bef0-882549304f48)

**Confusion Matrix for Table 3**
![CM_SOTA](https://github.com/AfsahS/Supervised-Contrastive-Ordinal-Loss-for-Ordinal-Regression/assets/52653609/0819902d-7735-4e15-9847-9c44f2443a26)



