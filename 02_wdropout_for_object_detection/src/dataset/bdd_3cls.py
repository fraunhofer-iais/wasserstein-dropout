import os
import json
from dataset.imdb import imdb
from utils.util import bbox_transform_inv, batch_iou


class BDD_3CLS(imdb):

  # image_set should be in ['train', 'val']
  def __init__(self, image_set, data_path, mc):
    super(BDD_3CLS, self).__init__('bdd_3cls', mc)

    self._data_root_path = data_path
    self._image_set = image_set
    self._images_dir = os.path.join(self._data_root_path, 'images', '100k', image_set)
    self._annotations_json = os.path.join(self._data_root_path, 'labels', 'detection20',
                                          'det_v2_{:s}_release.json'.format(image_set))

    assert os.path.isdir(self._images_dir), '{:s} is not a directory'.format(self._images_dir)
    assert os.path.isfile(self._annotations_json), '{:s} is not a file'.format(self._annotations_json)

    classes = set()
    with open(self._annotations_json) as f:
      annotations = json.load(f)

    self._image_idx_to_name = dict()
    for img_idx, annotation in enumerate(annotations):
      self._image_idx_to_name[img_idx] = annotation['name']
    
    self._classes = self.mc.CLASS_NAMES
    self._class_to_idx = {'truck': 2, 
                         'motorcycle': 2, 
                         'train': 3, 
                         'traffic light': 3, 
                         'traffic sign': 3, 
                         'pedestrian': 0,
                         'other vehicle': 2, 
                         'rider': 3, 
                         'bus': 2, 
                         'trailer': 2, 
                         'other person': 0, 
                         'bicycle': 1, 
                         'car': 2}

    non_ignore_classes = {'truck', 'motorcycle', 'pedestrian', 'other vehicle', 'bus', 'trailer', 'other person', 'bicycle', 'car'}

    self._rois = dict()
    self._image_idx = []
    for img_idx, annotation in enumerate(annotations):
  
      # ignore images that contain only labels from 'ignore' classes
      if annotation['labels'] is not None:
        img_label_classes = []
        for ann in annotation['labels']:
          img_label_classes.append(ann['category'])
        img_label_classes=set(img_label_classes)

        if not img_label_classes.intersection(non_ignore_classes):
          continue
                
      name = annotation['name']
      labels = annotation['labels']

      self._rois[img_idx] = []
      if labels is not None:
        self._image_idx.append(img_idx)
        for label in labels:

          cls_idx = self._class_to_idx[label['category'].strip().lower()]
          
          # ignore residual classes (mapped to 3)
          if cls_idx == 3:
            continue
            
          xmin, xmax = label['box2d']['x1'], label['box2d']['x2']
          ymin, ymax = label['box2d']['y1'], label['box2d']['y2']

          assert 0.0 <= xmin <= xmax, "Wrong bbox format; 0 <= xmin <= xmax not fulfilled"
          assert 0.0 <= ymin <= ymax, "Wrong bbox format; 0 <= ymin <= ymax not fulfilled"

          cx, cy, w, h = bbox_transform_inv([xmin, ymin, xmax, ymax])

          self._rois[img_idx].append([cx, cy, w, h, cls_idx, label['id'], img_idx, name])

    # batch reader
    self._shuffle_image_idx()
    self._cur_idx = 0

  def _image_path_at(self, idx):
    image_path = os.path.join(self._images_dir, self._image_idx_to_name[idx])
    assert os.path.exists(image_path), \
        'Image does not exist: {}'.format(image_path)
    return image_path