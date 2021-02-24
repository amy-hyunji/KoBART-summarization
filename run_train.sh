./prepare.sh
pip install -r requirements.txt
#python3 train.py  --gradient_clip_val 1.0 --max_epochs 50 --default_root_dir logs --gpus 1 --batch_size 4 
python3 train.py  --gradient_clip_val 1.0 --max_epochs 50 --default_root_dir logs  --accelerator 'dp' --gpus 2 --batch_size 4 
