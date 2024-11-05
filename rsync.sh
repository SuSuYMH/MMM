rsync -avP \
    --exclude '__pycache__/' \
    --exclude '*.py[cod]' \
    --exclude './output/*' \
    --exclude 'checkpoints' \
    --exclude 'glove' \
    --exclude 'dataset/HumanML3D' \
    --exclude 'dataset/KIT-ML' \
    --exclude 'output' \
    -e "ssh" A100:/data_ssd2/ymh/MMM /Users/su/Master-Project/MMM