##Setup Instructions:
#Create the data directory structure:
brain_tumor_data/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/

#Install required packages:
pip install torch torchvision torchaudio matplotlib seaborn scikit-learn pandas pillow

#Run the script:
python Squeezed-Eff-Net.py
