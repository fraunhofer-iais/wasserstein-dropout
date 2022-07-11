import os
import json
from dataset.imdb import imdb
from utils.util import bbox_transform_inv, batch_iou


class NUIMAGES(imdb):

  # image_set should be in ['train', 'val']
  def __init__(self, image_set, data_path, mc):
    super(NUIMAGES, self).__init__('nuimages', mc)
    
    self._data_root_path = data_path
    self._image_set = image_set
    self._images_dir = os.path.join(self._data_root_path, 'samples')
    self._annotations_json = os.path.join(self._data_root_path, 'v1.0-{:s}'.format(image_set), 'object_ann.json')
    self._sample_data_json = os.path.join(self._data_root_path, 'v1.0-{:s}'.format(image_set), 'sample_data.json')
    self._category_json = os.path.join(self._data_root_path, 'v1.0-{:s}'.format(image_set), 'category.json')
    
    assert os.path.isdir(self._images_dir), '{:s} is not a directory'.format(self._images_dir)
    assert os.path.isfile(self._annotations_json), '{:s} is not a file'.format(self._annotations_json)
    assert os.path.isfile(self._category_json), '{:s} is not a file'.format(self._category_json)
    
    classes = set()
    self._classes = mc.CLASS_NAMES

    with open(self._annotations_json) as f:
      annotations = json.load(f)

    with open(self._sample_data_json) as f:
      sample_data = json.load(f)
    
    with open(self._category_json) as f:
      obj_categories = json.load(f)

    self._image_token_to_name = dict()
    self._image_token_to_idx = dict()
    self._image_idx_to_name = dict()    
    for i, img in enumerate(sample_data):
       self._image_token_to_idx[img['token']] = i
       self._image_token_to_name[img['token']] = img['filename']
       self._image_idx_to_name[i] = img['filename']
    
    cat_token_to_cat = {}
    for cat in obj_categories:
      cat_token_to_cat[cat['token']] = cat['name']

    cat_to_red_cat = {
    'vehicle.car': 'car',
    'vehicle.truck': 'truck',
    'vehicle.bicycle': 'bicycle',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bus.rigid': 'vehicle_other',
    'vehicle.construction': 'vehicle_other',
    'vehicle.trailer': 'vehicle_other',
    'vehicle.bus.bendy': 'vehicle_other',
    'vehicle.emergency.police': 'vehicle_other',
    'vehicle.emergency.ambulance': 'vehicle_other',
    'vehicle.ego': 'vehicle_other',
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'human.pedestrian.personal_mobility': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.stroller': 'pedestrian',
    'human.pedestrian.wheelchair': 'pedestrian',
    'movable_object.trafficcone': 'movable_objects',
    'movable_object.barrier': 'movable_objects',
    'movable_object.debris': 'movable_objects',
    'movable_object.pushable_pullable': 'movable_objects',
    'static_object.bicycle_rack': 'movable_objects',
    'animal': 'movable_objects'
    }

    red_cats = ['pedestrian', 'bicycle', 'motorcycle', 'car', 'truck', 'vehicle_other', 'movable_objects']
    red_cat_to_idx = dict(zip(red_cats, xrange(len(red_cats))))
    
    self._rois = dict()
    self._image_idx = []
    for obj in annotations:

      img_idx = self._image_token_to_idx[obj['sample_data_token']]    
      name = self._image_token_to_name[obj['sample_data_token']]       
        
      if img_idx not in self._rois.keys():  
        self._rois[img_idx] = []
        self._image_idx.append(img_idx)

      xmin, ymin, xmax, ymax = obj['bbox']
      cx, cy, w, h = bbox_transform_inv([xmin, ymin, xmax, ymax])
      cls_idx = red_cat_to_idx[cat_to_red_cat[cat_token_to_cat[obj['category_token']]]]  
    
      self._rois[img_idx].append([cx, cy, w, h, cls_idx, name])

    # batch reader
    self._shuffle_image_idx()
    self._cur_idx = 0

  def _image_path_at(self, idx):
    image_path = os.path.join(self._images_dir, self._image_idx_to_name[idx].replace('samples/',''))
    assert os.path.exists(image_path), \
        'Image does not exist: {}'.format(image_path)
    return image_path
