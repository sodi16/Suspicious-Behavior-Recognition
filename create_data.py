
import os
import cv2
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
import sys
from xml.etree import ElementTree
import pylab as pl

# class that defines and loads the dataset
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


def load_frames_of_video(video_path, new_frame_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    count = 0
    if os.path.exists(new_frame_path):
        shutil.rmtree(new_frame_path)
        os.mkdir(new_frame_path)
    else:
        os.mkdir(new_frame_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(new_frame_path + r'\frame{:d}.jpg'.format(count), frame)
            count += 1
            cap.set(25, count)
        else:
            cap.release()
            cv2.destroyAllWindows()
            break


def run_from_video(frame_path, action_name, caviar_obj):
    img = None
    caviar_obj.load_dataset(frame_path, action_name, is_train=True)
    caviar_obj.prepare()
    num_imgs = len(os.listdir(frame_path))

    for id  in range(num_imgs):
        #if id < 100:
        f = os.path.join(frame_path, 'frame{0}.jpg'.format(id))
        image = pl.imread(f)
        mask, class_ids = caviar_obj.load_masks(id, image, action_name)

        if img is None:
            img = pl.imshow(mask)
        else:
            img.set_data(mask)
        pl.draw()
        pl.title('Frame '+ str(id))

        if id == 1000:
            break
            pl.close('all')

        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            pl.close('all')
            break


def run_from_video2(frame_path, action_name, caviar_obj):
    caviar_obj.load_dataset(frame_path, action_name, is_train=True)
    caviar_obj.prepare()
    num_frame = len(os.listdir(frame_path))
    X = []
    y = np.zeros([int((num_frame - 251)/3)+1, 2], dtype=object)
    #y = np.zeros([int((num_frame - 251)/2)+1, 2], dtype=object)

    for i, ids in enumerate(np.arange(250, num_frame - 1, 3)):
    #for i, ids in enumerate(np.arange(250, num_frame - 1, 2)):
        f = os.path.join(frame_path, 'frame{0}.jpg'.format(ids))
        image = pl.imread(f)
        #X[i] = image
        X.append(image)

        # extract the box and movements = Labels
        path = caviar_obj.image_info[ids]['annotation']
        box, mov = caviar_obj.extract_boxes(path, ids)
        print(i)

        if len(box) == 0:
            break
        box = np.array(box[0])
        mov = np.array(mov)
        temp = np.zeros([1, len(box)], dtype=object)

        if len(mov) > 0:
            for w in range(len(box)):
                temp[0, w] = mov[w]
            y[i, 0] = [z[0] for z in temp[0]]
            y[i, 1] = [z[1] for z in temp[0]]  
        else:
            y[i,:] = [],[]
    return X, y


def suspicious_behavior_labels(labels):
    very_suspicious_index = 2
    suspicious_index = 1
    not_suspicious_index = 0
    num_frames = len(labels)

    new_labels = np.array([],dtype=int)
    check_role = np.array([])
    check_context = np.array([])

    for i in range(num_frames):
        if labels[i,0] != [] and labels[i,1] != []:
            check_role = np.append(check_role, np.max(labels[i,0]))
            check_context = np.append(check_context, np.max(labels[i,1]))
        else:
            check_role = np.append(check_role, 0)
            check_context = np.append(check_context, 0)

    for i in range(num_frames):
        if check_role[i] == very_suspicious_index  or check_context[i] == very_suspicious_index:
            new_labels = np.append(new_labels, 2)
        elif check_role[i] == suspicious_index  or check_context[i] == suspicious_index:
            new_labels = np.append(new_labels, 1)
        else:
            new_labels = np.append(new_labels, 0)

    return new_labels.reshape(len(new_labels), 1)



if __name__ == '__main__':

    main_dir = r'C:\Users\so16s\AppData\Roaming\JetBrains\PyCharmCE2020.2\scratches\Caviar Dataset'

    #all action recognitions
    actions = ['Browse', 'Fight', 'Groups_Meeting', 'LeftBag', 'Rest', 'Walk', 'OneLeaveShop']
    print(actions)

    all_roles = {'fighters': 2, 'fighter': 2, 'leaving object': 2, 'browser': 1, 'browsers': 1, 'walkers': 0, 'meet': 0, 'meeters': 0, 'walker': 0}
    all_context = {'fighting': 2, 'leaving': 2, 'drop down': 2, 'browsing': 1, 'immobile': 0, 'walking': 0, 'meeting': 0, 'windowshop': 0, 'shop enter': 0, 'shop exit': 0, 'shop reenter': 0, 'none':0}

    #choose a action file 0-4
    id_dataset = int(input('Choose your action data from 0-{0} \n'.format(len(actions)-1)))
    action_dir = os.path.join(main_dir, actions[id_dataset])
    all_files_len= len(os.listdir(action_dir))


    train_testX, train_testy = [], []

    num_frames_train = 0
    num_frames = 0
    video_to_train = [1,2,3] #add all the video you want to train
    
    for id_vid in video_to_train:
    #for id_vid in range(1, all_files_len + 1):
        data_dir = os.path.join(action_dir, actions[id_dataset]) + str(id_vid)
        train_set = CaviarDataset()

        new_frames_file = main_dir + r'\new'  # video is converted to  images in this dir
        video_file = os.path.join(data_dir + r'\video', os.listdir(data_dir + r'\video')[0])

        load_frames_of_video(os.path.join(data_dir, video_file) , new_frames_file)

        X_frames, y_frames = run_from_video2(new_frames_file, actions[id_dataset], train_set)

        print('video number {0} \n X_frames = {1} \n y_frames = {2}'.format(id_vid, len(X_frames), len(y_frames)))

        train_testX.append(np.array(X_frames))
        train_testy.append(np.array(y_frames))
        num_frames += len(X_frames)


    X = np.concatenate((train_testX[0], train_testX[1], train_testX[2]), axis=0)
    y = np.concatenate((train_testy[0], train_testy[1], train_testy[2]), axis=0)
    y = suspicious_behavior_labels(y)

    with open(os.path.join(r'\{0}\X.npy'.format(actions[id_dataset])), 'wb') as f:
        np.save(f, X)
    f.close()
    with open(os.path.join(r'\{0}\y.npy'.format(actions[id_dataset])), 'wb') as f:
        np.save(f, y)
    f.close()





