import urllib.request
import json
import os
from tqdm import tqdm


def download_ctc_images(ctc_annotations_name='CTC_anns.json', dir_name='./images/'):
    """
    Downloading CTC_images from web using given annotation.json file
    :param ctc_annotations_name: information about CTC Dataset
    :param dir_name: New dir for images
    :return:
    """
    print("Reading annotations...")
    with open(ctc_annotations_name, 'r') as fp:
        ctc_anns = json.load(fp)

    print("Making new dir for CTC images...")
    os.mkdir(dir_name)

    print("Downloading CTC images from web to new dir")
    for item in tqdm(ctc_anns['images']):
        urllib.request.urlretrieve(item['coco_url'], dir_name + item['coco_url'].strip().split('/')[-1])
