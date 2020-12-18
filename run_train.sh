#./prepare.sh
#pip install -r requirements.txt
python train.py  --gradient_clip_val 1.0 --max_epochs 50 --default_root_dir logs  --accelerator 'dp' --log_gpu_memory 'all' --gpus 3 --batch_size 4 --weights_save_path "./ckpt"
