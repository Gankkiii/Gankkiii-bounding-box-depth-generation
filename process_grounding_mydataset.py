import torch
# from ldm.modules.encoders.modules import FrozenCLIPEmbedder
# from ldm.modules.encoders.modules import BERTEmbedder
from transformers import CLIPProcessor, CLIPModel
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
import argparse
import os
import math
import clip
from PIL import Image
from torchvision import transforms
import multiprocessing
from zipfile import ZipFile
from io import BytesIO
import base64

def split_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def clean_annotations(annotations):
    for anno in annotations:
        anno.pop("segmentation", None)
        anno.pop("area", None)
        anno.pop("iscrowd", None)


def project(x, projection_matrix):
    """
    x (Batch*768) should be the penultimate feature of CLIP (before projection)
    projection_matrix (768*768) is the CLIP projection matrix, which should be weight.data of Linear layer
    defined in CLIP (out_dim, in_dim), thus we need to apply transpose below.
    this function will return the CLIP feature (without normalziation)
    """
    return x @ torch.transpose(projection_matrix, 0, 1)


def inv_project(y, projection_matrix):
    """
    y (Batch*768) should be the CLIP feature (after projection)
    projection_matrix (768*768) is the CLIP projection matrix, which should be weight.data of Linear layer
    defined in CLIP (out_dim, in_dim).
    this function will return the CLIP penultimate feature.

    Note: to make sure getting the correct penultimate feature, the input y should not be normalized.
    If it is normalized, then the result will be scaled by CLIP feature norm, which is unknown.
    """
    return y @ torch.transpose(torch.linalg.inv(projection_matrix), 0, 1)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #


class Base():
    def __init__(self, image_root):
        self.image_root = image_root
        self.use_zip = True if image_root[-4:] == ".zip" else False
        if self.use_zip:
            self.zip_dict = {}

        # This is CLIP mean and std
        # Since our image is cropped from bounding box, thus we directly resize to 224*224 without center_crop to keep obj whole information.
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])

    def fetch_zipfile(self, ziproot):
        pid = multiprocessing.current_process().pid  # get pid of this process.
        if pid not in self.zip_dict:
            self.zip_dict[pid] = ZipFile(ziproot)
        zip_file = self.zip_dict[pid]
        return zip_file

    def fetch_image(self, file_name):
        if self.use_zip:
            zip_file = self.fetch_zipfile(self.image_root)
            image = Image.open(BytesIO(zip_file.read(file_name))).convert('RGB')
        else:
            image = Image.open(os.path.join(self.image_root, file_name)).convert('RGB')
        return image



class GroundedTextImageDataset_Grounding(Base):
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            loaded_dict_list = json.load(f)
        self.image = {}
        self.caption = {}
        self.annotations = []
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])

        for d in loaded_dict_list:
            image_data_decoded = base64.b64decode(d["image"])
            image = Image.open(BytesIO(image_data_decoded))
            self.image[d["data_id"]] = image
            self.caption[d["data_id"]] = d["caption"]
            self.annotations += d["annos"]

    def __getitem__(self, index):

        anno = self.annotations[index]
        anno_id = anno["id"]
        X, Y, W, H = anno['bbox']

        caption = self.caption[anno["data_id"]]
        image = self.image[anno["data_id"]]
        image_crop = self.preprocess(image.crop((X, Y, X + W, Y + H)).resize((224, 224), Image.BICUBIC))

        positive = ''
        for (start, end) in anno['tokens_positive']:
            positive += caption[start:end]
            positive += ' '
        positive = positive[:-1]

        return {'positive': positive, 'anno_id': anno_id, 'image_crop': image_crop}

    def __len__(self):
        return len(self.annotations)


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =#


@torch.no_grad()
def fire_clip_before_after(loader, folder):
    """
    This will save CLIP feature before/after projection.

    before projection text feature is the one used by stable-diffsuion.
    For before_projection, its feature is unmormalized.
    For after_projection, which is CLIP aligned space, its feature is normalized.

    You may want to use project / inv_project to project image feature into CLIP text space. (Haotian's idea)
    """
    version = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(version).cuda()
    processor = CLIPProcessor.from_pretrained(version)
    # projection_matrix = torch.load('projection_matrix').cuda()

    os.makedirs(os.path.join(folder, 'text_features_before'), exist_ok=True)
    os.makedirs(os.path.join(folder, 'text_features_after'), exist_ok=True)
    os.makedirs(os.path.join(folder, 'image_features_before'), exist_ok=True)
    os.makedirs(os.path.join(folder, 'image_features_after'), exist_ok=True)

    for batch in tqdm(loader):

        inputs = processor(text=batch['positive'], return_tensors="pt", padding=True)
        inputs['input_ids'] = inputs['input_ids'].cuda()
        inputs['pixel_values'] = batch['image_crop'].cuda()  # we use our own preprocessing without center_crop
        inputs['attention_mask'] = inputs['attention_mask'].cuda()
        outputs = model(**inputs)

        text_before_features = outputs.text_model_output.pooler_output  # before projection feature
        text_after_features = outputs.text_embeds  # normalized after projection feature (CLIP aligned space)

        image_before_features = outputs.vision_model_output.pooler_output  # before projection feature
        image_after_features = outputs.image_embeds  # normalized after projection feature (CLIP aligned space)

        for idx, text_before, text_after, image_before, image_after in zip(batch["anno_id"], text_before_features,
                                                                           text_after_features, image_before_features,
                                                                           image_after_features):
            save_name = os.path.join(folder, 'text_features_before', str(int(idx)))
            torch.save(text_before.clone().cpu(), save_name)

            save_name = os.path.join(folder, 'text_features_after', str(int(idx)))
            torch.save(text_after.clone().cpu(), save_name)

            save_name = os.path.join(folder, 'image_features_before', str(int(idx)))
            torch.save(image_before.clone().cpu(), save_name)

            save_name = os.path.join(folder, 'image_features_after', str(int(idx)))
            torch.save(image_after.clone().cpu(), save_name)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default=r"D:\jupyter\project\training_data.json",
                        help="")
    parser.add_argument("--folder", type=str, default=r"D:\project\gligen dataset\out", help="")
    args = parser.parse_args()



    dataset = GroundedTextImageDataset_Grounding(args.json_path)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4, drop_last=False)
    os.makedirs(args.folder, exist_ok=True)
    fire_clip_before_after(loader, args.folder)
