import csv
import os
import shutil
from pathlib import Path

def move_error_images(csv_path, target_dir):
    # Ensure target directory exists
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    moved_count = 0
    not_found_count = 0
    error_tag_count = 0
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    with open(csv_path, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            path_str = row.get('path')
            tags_str = row.get('tags', '')
            
            if not path_str:
                continue
                
            tags = [t.strip() for t in tags_str.split(',')]
            
            if 'error' in tags:
                error_tag_count += 1
                source_file = Path(path_str)
                
                if source_file.exists():
                    dest_file = target_path / source_file.name
                    
                    try:
                        # Move the file
                        shutil.move(str(source_file), str(dest_file))
                        print(f"Moved: {source_file.name}")
                        moved_count += 1
                    except Exception as e:
                        # If permission error occurs, report it clearly
                        if isinstance(e, PermissionError):
                            print(f"Permission Denied: Unable to move {source_file}. Please check directory permissions.")
                        else:
                            print(f"Failed to move {source_file}: {e}")
                else:
                    # Check if it was already moved
                    if (target_path / source_file.name).exists():
                        # Already moved in a previous run or by other means
                        pass
                    else:
                        print(f"File not found: {path_str}")
                        not_found_count += 1
                    
    print("\nSummary:")
    print(f"Total rows with 'error' tag: {error_tag_count}")
    print(f"Successfully moved: {moved_count}")
    print(f"Files not found/skipped: {not_found_count}")

if __name__ == "__main__":
    PROJECT_ROOT = "/home/master-user/Desktop/Engineering_Image_Retrieval_System_Training"
    CSV_FILE = os.path.join(PROJECT_ROOT, "outputs/manual_tags.csv")
    TARGET_DIR = os.path.join(PROJECT_ROOT, "data/converted_images_error")
    move_error_images(CSV_FILE, TARGET_DIR)
