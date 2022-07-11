import os
import json
from dataset.imdb import imdb
from utils.util import bbox_transform_inv, batch_iou


class A2D2(imdb):

  # image_set should be in ['train', 'val']
  def __init__(self, image_set, data_path, mc):
    super(A2D2, self).__init__('a2d2', mc)
    
    self._data_root_path = data_path
    self._image_set = image_set

    dict_train_val_seq_split = {
    "train": ["20180807_145028", "20180925_135056", "20181108_091945", 
              "20181204_154421", "20181204_135952", "20181108_084007",
              "20181107_132730", "20181108_103155", "20181204_170238",
              "20181016_082154", "20181107_133258", "20181108_123750",
              "20180925_112730", "20181016_095036", "20181107_133445",
              "20181108_141609", "20180925_124435", "20181016_125231"
              ],
    "val": ["20181008_095521", "20181107_132300", "20180925_101535", "20181204_191844"], 
    "test": ["20180810_142822"], 
    "mini": ["mini"],
    }
    
    seq_list = sorted(dict_train_val_seq_split[image_set])

    self._classes = self.mc.CLASS_NAMES
    self._class_idx_map = dict(zip(self._classes, xrange(len(self._classes))))

    self._rois = dict()
    self._image_idx = []

    self._image_idx_to_name = dict()
    img_idx = 0
    for seq in seq_list:
      img_ann_filenames = [f for f in os.listdir(os.path.join(data_path,seq,'label2D','cam_front_center')) if f.endswith('.json')]
      img_ann_filenames = sorted(img_ann_filenames)  
        
      for img_ann_filename in img_ann_filenames:
        self._image_idx_to_name[img_idx] = img_ann_filename

        with open(os.path.join(data_path, seq, 'label2D', 'cam_front_center', img_ann_filename)) as file:
          annotations = json.load(file)   
  
        object_ids = annotations.keys()
  
        if object_ids:
          self._rois[img_idx] = []
          self._image_idx.append(img_idx)
        
        for object_id in object_ids:
          xmin, ymin, xmax, ymax = annotations[object_id]['2d_bbox']

          cx, cy, w, h = bbox_transform_inv([xmin, ymin, xmax, ymax])
  
          cls_idx = self._class_idx_map[annotations[object_id]['class'].lower()]
  
          self._rois[img_idx].append([cx, cy, w, h, cls_idx])
    
        img_idx += 1

    # batch reader
    self._shuffle_image_idx()
    self._cur_idx = 0

  def _image_path_at(self, idx):

    ann_filename = self._image_idx_to_name[idx]
    datetime = ann_filename.split('_')[0]
    seq = datetime[:8]+'_'+datetime[8:]
    img_filename = ann_filename.replace('label2D','camera').replace('.json','.png')
    
    image_path = os.path.join(self._data_root_path,seq,'camera','cam_front_center',img_filename)
    assert os.path.exists(image_path), \
        'Image does not exist: {}'.format(image_path)
    return image_path