train_vimeo: 
	python3 train_v5_vimeo.py --data_root /home/ortak/mughees/datasets/vimeo_triplet
validate_vimeo:
	python3 Vimeo90K_val.py --data_root /home/ortak/mughees/datasets/vimeo_triplet
clean:
	rm -rf __pycache__ ./model/__pycache_ project.mp4 logs/* model_dict/* train_log/
