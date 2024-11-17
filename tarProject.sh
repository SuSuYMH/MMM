tar --exclude '__pycache__/' \
    --exclude '*.py[cod]' \
    --exclude './output/*' \
    --exclude 'checkpoints' \
    --exclude 'glove' \
    --exclude 'dataset/HumanML3D' \
    --exclude 'dataset/KIT-ML' \
    --exclude 'output' \
    -czvf MMM.tar.gz 

# 要到MMM的上一级目录
tar --exclude='dataset/HumanML3D' --exclude='output' --exclude='checkpoints' --exclude='qwen' -czvf MMM-11.17.tar.gz MMM