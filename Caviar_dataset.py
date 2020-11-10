import os
import cv2
import numpy as np
import shutil
from sklearn.model_selection import train_test_split

import sys
from xml.etree import ElementTree
import pylab as pl


class CaviarDataset():

    def __init__(self, class_map=None):
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}
        #Mapping from source class and image IDs to internal IDs


    def add_class(self, source, class_id, class_name):
        for info in self.class_info:
            if info['source'] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)


    def prepare(self, class_map=None):

        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)

        # Mapping from source class and image IDs to internal IDs


        # Map sources to class_ids they support
        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}
        # Loop over datasets
        for source in self.sources:
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)

    # load the dataset definitions
    def load_dataset(self, dataset_dir, obj_class, is_train=True):
        # define one class
        self.add_class(dataset_dir, 1, obj_class)
        annotations_dir = data_dir + fr'\annots'

        # find all images
        for im_id, filename in enumerate(os.listdir(dataset_dir)):
            list = os.listdir(dataset_dir)

            img_path = os.path.join(data_dir, filename)

            annots = os.path.join(annotations_dir, os.listdir(annotations_dir)[0])
            self.add_image('dataset', image_id=im_id, path=img_path, annotation=annots)


    # load the masks for an image
    def load_masks(self, image_id, image, obj_class):
        font = cv2.FONT_HERSHEY_SIMPLEX
        #path = label_dir  # TO CHANGE
        path = self.image_info[image_id]['annotation']
        # load XML
        boxes, movements = self.extract_boxes(path, image_id)
        # create masks
        class_ids = list()
        new_masks = image
        colors = [[0,0,255],[0,255,0],[255,0,0],[255,255,0],[0,255,255],[255,0,255]]
        for i in range(len(boxes[0])):
            box = boxes[0][i].astype(int)
            mov = movements[i]
            y_2 = int(np.abs(box[3] + (box[0]/2)))
            y_1 = int(np.abs(box[3] - (box[0]/2)))
            x_2 = int(np.abs(box[2] + (box[1]/2)))
            x_1 = int(np.abs(box[2] - (box[1]/2)))
            num_col = i % len(colors)
            new_masks = cv2.rectangle(new_masks,(x_1,y_1),(x_2, y_2),colors[num_col],1)
            #img, text, location, font, fontsize, color, linetype
            x1 = x_1 - 1
            y = y_2 - 1
            cv2.putText(img=new_masks,text='{0}'.format(mov[0]), org=(x1 ,y_1), fontFace=font ,fontScale=0.3, color=colors[i], lineType=2)
            cv2.putText(img=new_masks,text='{0}'.format(mov[3]), org=(x1, y_2), fontFace=font, fontScale=0.3,color=colors[i], lineType=2)
            class_ids.append(self.class_names.index(obj_class))
        return new_masks, np.asarray(class_ids, dtype='int32')


    # function to extract bounding boxes from an annotation file
    def extract_boxes(self, filename, im_id):
        # load and parse the file
        tree = ElementTree.parse(filename)
        root = tree.getroot()
        # extract each bounding box
        boxes, all_labels = [], []
        for f, frames in enumerate(root.findall('frame')):
            xc, yc, h, w, sits, roles, movements, contexts = [], [], [], [], [], [], [], []
            if f == im_id:
                for box in frames.findall('.//box'):
                    xc.append(box.get('xc'))
                    yc.append(box.get('yc'))
                    h.append(box.get('h'))
                    w.append(box.get('w'))
                for role in frames.findall('.//role'):
                    roles.append(role.text)
                for context in frames.findall('.//context'):
                    contexts.append(context.text)

                coors = np.ones(shape=(len(xc), 4))
                a, b = coors.shape
                for i in range(a):
                    coors[i, :] = [h[i], w[i], xc[i], yc[i]]
                    #all_labels.append([all_roles[roles[i]], sits[i], movements[i], all_context[contexts[i]]])
                    all_labels.append([all_roles[roles[i]], all_context[contexts[i]]])
                boxes.append(coors)
        return boxes, all_labels
