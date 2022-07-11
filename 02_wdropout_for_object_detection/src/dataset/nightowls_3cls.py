import os
import json
from dataset.imdb import imdb
from utils.util import bbox_transform, bbox_transform_inv, batch_iou


class NIGHTOWLS_3CLS(imdb):

  # image_set should be in ['train', 'val']
  def __init__(self, image_set, data_path, mc):
    super(NIGHTOWLS_3CLS, self).__init__('nightowls_3cls', mc)

    if image_set == 'train':
        image_set = 'training'
    elif image_set == 'val':
        image_set = 'validation'
    
    self._data_root_path = data_path
    self._image_set = image_set
    self._images_dir = os.path.join(self._data_root_path, 'nightowls_'+image_set)
    self._annotations_json = os.path.join(self._data_root_path, 'nightowls_{:s}.json'.format(image_set))

    assert os.path.isdir(self._images_dir), '{:s} is not a directory'.format(self._images_dir)
    assert os.path.isfile(self._annotations_json), '{:s} is not a file'.format(self._annotations_json)

    classes = set()
    with open(self._annotations_json) as f:
      annotations = json.load(f)

    res_x = annotations['images'][0]['width']
    res_y = annotations['images'][0]['height']
    
    self._image_idx_to_name = dict()
    for annotation in annotations['images']:
      self._image_idx_to_name[annotation['id']] = annotation['file_name']
    
    dict_img_idx_list_idx = {}
    for idx, image in enumerate(annotations['images']):
      dict_img_idx_list_idx[image['id']] = idx
    
    dict_pose_idx_pose = {}
    for pose in annotations['poses']:
      dict_pose_idx_pose[pose['id']] = str(pose['name'])
    dict_pose_idx_pose[5] = 'not_documented'
    
    for annotation in annotations['annotations']:
          classes = classes.union({annotation['category_id']})
    #print(classes)
            
    if len(classes.intersection(set(self.mc.CLASS_NAMES))) == len(classes):
      print("WARNING: Class names given in mc does not match class names of read files")
    self._classes = self.mc.CLASS_NAMES
    #self._class_to_idx = dict(zip(self.classes, xrange(self.num_classes)))

    self._rois = dict()
    self._image_idx = []
    for annotation in annotations['annotations']:

      img_idx = annotation['image_id'] 
      name = self._image_idx_to_name[img_idx]        

      img_attributes = [img_idx, name]
      img_list_idx = dict_img_idx_list_idx[img_idx]
      for attribute in ['daytime', 'timestamp', 'recordings_id']:
        img_attributes.append(annotations['images'][img_list_idx][attribute])
      
      # exclude all 'ignore_area' labels for 3cls setting
      if annotation['category_id']==4:
            continue

      cx, cy, w, h = annotation['bbox']
      
      if (w<0) or (h<0):
        continue
    
      xmin, ymin, xmax, ymax = bbox_transform([cx, cy, w, h])
        
      if (xmin > res_x) or (xmax < 0):
        continue
      elif (ymin > res_y) or (ymax < 0):
        continue

      if img_idx not in self._rois.keys():  
        self._rois[img_idx] = []
        self._image_idx.append(img_idx)
        
      xmin = max(0, xmin)
      ymin = max(0, ymin)
      xmax = min(res_x, xmax)
      ymax = min(res_y, ymax)
            
      cx, cy, w, h = bbox_transform_inv([xmin, ymin, xmax, ymax])
    
      cls_idx = annotation['category_id']-1

      pose = dict_pose_idx_pose[annotation['pose_id']]
    
      self._rois[img_idx].append([cx, cy, w, h, cls_idx, pose] + [annotation[key] for key in ['area','difficult','id','ignore','occluded','tracking_id','truncated']] + img_attributes)

    # batch reader
    self._shuffle_image_idx()
    self._cur_idx = 0

  def _image_path_at(self, idx):
    image_path = os.path.join(self._images_dir, self._image_idx_to_name[idx])
    assert os.path.exists(image_path), \
        'Image does not exist: {}'.format(image_path)
    return image_path
