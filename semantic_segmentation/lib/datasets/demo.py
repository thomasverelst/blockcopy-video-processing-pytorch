import os
import glob
import re
import logging
# import cv2
from PIL import Image

def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower()
            for text in _nsre.split(s)]    

class DemoImageDataset():
    '''
    Demo loader
    Loads all images in folder and subfolders of given root folder
    
    Files are gathered as paths and then sorted with natural sort
    '''
    def __init__(self, root: str, transform=None) -> None:
        self.root = root
        self.transform = transform
        
        files = glob.glob(os.path.join(root,'**/*'), recursive=True)
        # files = [files]
        files = [f for f in files if f.endswith(('.png','.jpg'))]
        files = sorted(files, key=natural_sort_key)       
         
        self.image_paths = files
        
        logging.info(f'DemoDataset: loaded {len(self.image_paths)} images from {root}')
            
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple[torch.Tensor, Optional[torch.Tensor], dict] : tuple of image, label and meta
        """
        path = self.image_paths[index]
        meta = {
            'path': path,
            'relpath': os.path.relpath(path, self.root)
        }
        
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image, _ = self.transform(image, None)
        image = image.unsqueeze(0) # 1-image clips
        return image, False, meta

        