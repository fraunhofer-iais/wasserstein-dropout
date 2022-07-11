import os
import json
from dataset.imdb import imdb
from utils.util import bbox_transform_inv, batch_iou


class SYNSCAPES(imdb):

  # image_set should be in ['train', 'val']
  def __init__(self, image_set, data_path, mc):
    super(SYNSCAPES, self).__init__('synscapes', mc)
    
    self._data_root_path = data_path
    self._image_set = image_set
    self._images_dir = os.path.join(self._data_root_path, 'img', 'rgb')

    if image_set == 'train':
        with open(os.path.join(data_path, 'train_idxs.txt')) as file:
          img_idxs = json.load(file)
    elif image_set == 'val':
        with open(os.path.join(data_path, 'val_idxs.txt')) as file:
          img_idxs = json.load(file)
    
    assert os.path.isdir(self._images_dir), '{:s} is not a directory'.format(self._images_dir)

    classes = [24, 26, 27, 28, 31, 32, 33]
    self._class_idx_map = dict(zip(classes, xrange(len(classes))))
    self._classes = self.mc.CLASS_NAMES

    self._rois = dict()
    self._image_idx = []

    for img_idx in img_idxs:
      with open(os.path.join(data_path, 'meta', str(img_idx)+'.json')) as file:
        annotations = json.load(file)   

      scene_annotations = annotations['scene'].values()

      object_ids = annotations['instance']['bbox2d'].keys()
    
      if object_ids:
        self._rois[img_idx] = []
        self._image_idx.append(img_idx)
    
      for object_id in object_ids:
        object_bbox_coords = annotations['instance']['bbox2d'][object_id]
        xmin_rel, ymin_rel, xmax_rel, ymax_rel = [object_bbox_coords[key]for key in ['xmin','ymin','xmax','ymax']]
       
        w = float(annotations['camera']['intrinsic']['resx'])
        h = float(annotations['camera']['intrinsic']['resy'])
        
        xmin = max(0, xmin_rel * w)
        ymin = max(0, ymin_rel * h)
        xmax = min(w, xmax_rel * w)
        ymax = min(h, ymax_rel * h)
        
        cx, cy, w, h = bbox_transform_inv([xmin, ymin, xmax, ymax])
            
        truncation = annotations['instance']['truncated'][object_id]
        occlusion = annotations['instance']['occluded'][object_id]
    
        cls_idx_raw = annotations['instance']['class'][object_id]
        cls_idx = self._class_idx_map[cls_idx_raw]

        self._rois[img_idx].append([cx, cy, w, h, cls_idx, truncation, occlusion]+scene_annotations)

    # batch reader
    self._shuffle_image_idx()
    self._cur_idx = 0

  def _image_path_at(self, idx):
    image_path = os.path.join(self._images_dir, str(idx)+'.png')
    assert os.path.exists(image_path), \
        'Image does not exist: {}'.format(image_path)
    return image_path
