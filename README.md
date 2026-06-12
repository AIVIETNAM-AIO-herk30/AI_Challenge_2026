#  Challenge 2026

Task: 

## Directory Structure

```text
├── data/                   # Data folder
│   ├── raw/                # Raw data
│   └── processed/          # Data after processed and augmented
├── notebooks/              # EDA & Prototype
├── src/                    # 
│   ├── data_loader.py      # Data pipeline & preprocessing
│   ├── model.py            # Neural network architectures
│   ├── train.py            # Training pipeline, loss & optimizer
│   └── inference.py        # Inference, evaluation & submission
├── weights/                # Trained model checkpoints
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation     
```

## Environment Setup
Cloning Project
```
git clone [https://github.com/AIVIETNAM-AIO-herk30/AI_Challenge_2026.git](https://github.com/AIVIETNAM-AIO-herk30/AI_Challenge_2026.git)
cd AI_Challenge_2026
``` 
Setup environment
```
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate   # Windows
```
Installing dependencies
```
pip install --upgrade pip
pip install -r requirements.txt
```

## Pipeline Execution
1. Data Preprocessing
```
python src/data_loader.py --input_dir data/raw/ --output_dir data/processed/
```
2. Model Training
```
python src/train.py --data_dir data/processed/ --epochs 50 --batch_size 32 --learning_rate 1e-4
```
3. Inference Submission
``` 
python src/inference.py --test_dir data/raw/test/ --weights weights/best_model.pth --out_file outputs/submission.csv
```

## Team Members
| No. | Name |
| :---: | :--- |
| 1 | Lê Nguyên Khôi |
| 2 | Phạm Viết Trường |
| 3 | Trương Hoàng Thông |
| 4 | Phạm Hữu Huy |