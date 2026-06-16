sudo apt update

sudo apt install unzip

unzip data/PDF.zip -d ./data

uv run python src/pdf_to_image2.py
<!-- uv run python src/pdf_to_image2.py --root_dir ./data/PDF --output_dir ./data/engineering_images_100dpi --dpi 100 --max_workers 20 -->


uv run python src/image_preprocessing_batch_multiprocess2.py --input_dir ./data/engineering_images_100dpi --output_root ./results/batch/engineering_images_100dpi

uv run python src/split_dataset.py

uv run python src/model/simsiam2_training.py
