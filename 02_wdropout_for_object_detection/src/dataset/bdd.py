import os
import json
from dataset.imdb import imdb
from utils.util import bbox_transform_inv, batch_iou


class BDD(imdb):

  # image_set should be in ['train', 'val']
  def __init__(self, image_set, data_path, mc):
    super(BDD, self).__init__('bdd', mc)

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
      if annotation['labels'] is not None:
          classes.union({label['category'].strip().lower() for label in annotation['labels']})

    if len(classes.intersection(set(self.mc.CLASS_NAMES))) < len(classes):
      print("WARNING: Class names given in mc does not match class names of read files")
    self._classes = self.mc.CLASS_NAMES
    self._class_to_idx = dict(zip(self.classes, xrange(self.num_classes)))

    self._rois = dict()
    self._image_idx = []
    for img_idx, annotation in enumerate(annotations):

      name = annotation['name']
      labels = annotation['labels']

      self._rois[img_idx] = []
      if labels is not None:
        self._image_idx.append(img_idx)
        for label in labels:
          xmin, xmax = label['box2d']['x1'], label['box2d']['x2']
          ymin, ymax = label['box2d']['y1'], label['box2d']['y2']

          assert 0.0 <= xmin <= xmax, "Wrong bbox format; 0 <= xmin <= xmax not fulfilled"
          assert 0.0 <= ymin <= ymax, "Wrong bbox format; 0 <= ymin <= ymax not fulfilled"

          cx, cy, w, h = bbox_transform_inv([xmin, ymin, xmax, ymax])
          cls_idx = self._class_to_idx[label['category'].strip().lower()]

          self._rois[img_idx].append([cx, cy, w, h, cls_idx, label['id'], img_idx, name])

    # batch reader
    self._shuffle_image_idx()
    self._cur_idx = 0

  def _image_path_at(self, idx):
    image_path = os.path.join(self._images_dir, self._image_idx_to_name[idx])
    assert os.path.exists(image_path), \
        'Image does not exist: {}'.format(image_path)
    return image_path
