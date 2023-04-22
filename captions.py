import os
import json

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class CaptionImageDataset(Dataset):
    def __init__(self, image_paths, seen_caption_paths, unseen_caption_path, transform=None):
        self.image_paths = image_paths
        self.seen_caption_paths = seen_caption_paths
        self.unseen_caption_path = unseen_caption_path
        self.label = []
        self.transform = transform

    def __getitem__(self, index):
        # Load image
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Load caption
        caption_path = self.seen_caption_paths[index]
        with open(caption_path, 'r') as f:
            caption = f.read()
        
        # Load unseen caption
        unseen_caption_path = self.unseen_caption_path[index]
        with open(unseen_caption_path, 'r') as f:
            caption = f.read()

        return image, caption

    def __len__(self):
        return len(self.image_paths)

class CaptionImageGenerator:
    def __init__(self, image_dir, seen_caption_dir,unseen_caption_dir, batch_size, transform=None):
        self.image_dir = image_dir
        self.seen_caption_dir = seen_caption_dir
        self.unseen_caption_dir = unseen_caption_dir
        self.batch_size = batch_size
        self.transform = transform

    def __iter__(self):
        image_paths = [os.path.join(self.image_dir, filename) for filename in os.listdir(self.image_dir)]
        seen_caption_paths = [os.path.join(self.seen_caption_dir, filename) for filename in os.listdir(self.seen_caption_dir)]
        unseen_seen_caption_paths = [os.path.join(self.unseen_caption_dir, filename) for filename in os.listdir(self.unseen_caption_dir)]
        dataset = CaptionImageDataset(image_paths=image_paths, seen_caption_paths=seen_caption_paths,unseen_caption_path=unseen_seen_caption_paths, transform=self.transform)
        dataloader = DataLoader(dataset)
        return iter(dataloader)


def create_json_file(image_folder, caption_folder, json_file):
    # Get list of image and caption files
    image_files = sorted(os.listdir(image_folder))
    caption_files = sorted(os.listdir(caption_folder))

    # Check that the number of image and caption files match
    if len(image_files) != len(caption_files):
        print("Error: Number of image and caption files do not match")
        return

    # Create dictionary to store image and caption data
    data_dict = {}
    for i in range(100):
        # Get image and caption file paths
        image_path = os.path.join(image_folder, image_files[i])
        caption_path = os.path.join(caption_folder, caption_files[i])

        # Read in caption data
        with open(caption_path, 'r') as f:
            caption = f.read().strip()

        # Add image and caption data to dictionary
        data_dict[image_files[i]] = {
            "image_path": image_path,
            "caption": caption
        }

    # Write dictionary to JSON file
    with open(json_file, 'w') as f:
        json.dump(data_dict, f)


# image_dir = 'data/train2017/images'
# caption_dir = 'data/train2017/captions'
# output_path = 'data.json'

# create_json_file(image_dir, caption_dir, output_path)
