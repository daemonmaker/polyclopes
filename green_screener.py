import string
import os
import os.path as op
import math
import copy
import json
from PIL import Image
import numpy as np
from scipy import ndimage


GREEN = (0, 177, 64)
NUM_CHANNELS = 3
MAX_BACKGROUND_WIDTH = 244
MAX_BACKGROUND_HEIGHT = 244
MAX_OBJECT_WIDTH = 200
MAX_OBJECT_HEIGHT = 200
RANDOM_SAMPLES_PER_OBJECT = 50


class Coords(object):
    def __init__(self, coords):
        self.x = coords[0]
        self.y = coords[1]

    def __str__(self):
        return "({}, {})".format(self.x, self.y)

    def __repr__(self):
        return str(self)


class BoundingBox(object):
    def __init__(self, coords):
        self.top_left = Coords(coords[:3])
        self.bottom_right = Coords(coords[2:])

    def width(self):
        return math.fabs(self.bottom_right.x - self.top_left.x)

    def height(self):
        return math.fabs(self.bottom_right.y - self.top_left.y)

    def __str__(self):
        return "top_left: {}\tbottom_right: {}\twidth: {}\t height: {}".format(
            self.top_left,
            self.bottom_right,
            self.width(),
            self.height()
        )

    def __repr(self):
        return str(self)


class Subject(object):
    def __init__(self, image_directory, image_name, label_name, label_id):
        self.image_direcory = image_directory
        self.image_name = image_name
        self.label_name = label_name
        self.label_id = label_id
        image_location = op.join(self.image_direcory, self.image_name)
        self.original_image = Image.open(image_location)
        self._extract_object()

    def _extract_object(self):
        img = self.original_image
        binary_map = np.zeros((img.width, img.height))
        pixels = img.load()
        for xdx in range(img.width):
            for ydx in range(img.height):
                current_pixel = pixels[xdx, ydx]
                if not (current_pixel[0] == GREEN[0] and current_pixel[1] == GREEN[1] and current_pixel[2] == GREEN[2]):
                    binary_map[xdx, ydx] = 1

        binary_map_image = Image.fromarray(255*binary_map.T).convert('L')
        binary_map = binary_map.astype('int')
        img = img.convert('RGBA')
        img.putalpha(binary_map_image)
        self.img = img
        objects = ndimage.find_objects(binary_map)

        top = objects[0][1].start
        left = objects[0][0].start
        right = objects[0][0].stop
        bottom = objects[0][1].stop
        self.object_image = img.crop((left, top, right, bottom))


class Sample(object):
    def __init__(self, subject, background, scale=1, draw_bounding_box=False):
        self.subject = subject

        object_image = copy.deepcopy(subject.object_image)

        # If the object is too big then scale it such that the largest size is the max possible size
        if object_image.height > MAX_OBJECT_HEIGHT or object_image.width > MAX_OBJECT_WIDTH:
            if object_image.height > object_image.width:
                scaler = MAX_OBJECT_HEIGHT / object_image.height
            else:
                scaler = MAX_OBJECT_WIDTH / object_image.width

            new_height = int(object_image.height*scaler)
            new_width = int(object_image.width*scaler)
            object_image = object_image.resize((new_width, new_height), Image.BILINEAR)

        if 0 < scale < 1:
            new_width = int(object_image.width*scale)
            new_height = int(object_image.height*scale)
            object_image = object_image.resize((new_width, new_height), resample=Image.BILINEAR)

        width = object_image.width
        height = object_image.height
        left = np.random.randint(0, background.width - object_image.width)
        top = np.random.randint(0, background.height - object_image.height)
        self.bounding_box = BoundingBox((left, top, left+width, top+height))

        background = background.convert('RGBA')
        background.paste(object_image, (left, top), object_image)

        if draw_bounding_box:
            pixels = background.load()
            for xdx in range(self.bounding_box.top_left.x, self.bounding_box.bottom_right.x+1):
                pixels[xdx, self.bounding_box.top_left.y] = (255, 0, 0)
                pixels[xdx, self.bounding_box.bottom_right.y] = (255, 0, 0)

            for ydx in range(self.bounding_box.top_left.y, self.bounding_box.bottom_right.y+1):
                pixels[self.bounding_box.top_left.x, ydx] = (255, 0, 0)
                pixels[self.bounding_box.bottom_right.x, ydx] = (255, 0, 0)

        self.image = background


def load_backgrounds(backgrounds_dir):
    backgrounds = []
    with os.scandir(backgrounds_dir) as it:
        for entry in it:
            if entry.is_file() and entry.name[-4:] in ('.png', '.jpg'):
                print('Processing background: ', entry.name)
                background = Image.open(op.join(backgrounds_dir, entry.name))
                if background.width < background.height:
                    scaler = MAX_BACKGROUND_WIDTH / background.width
                else:
                    scaler = MAX_BACKGROUND_HEIGHT / background.height
                new_width = int(background.width*scaler)
                new_height = int(background.height*scaler)
                background = background.resize((new_width, new_height), Image.BILINEAR)
                backgrounds.append(background)
    return backgrounds


def crop_background(background):
    furthest_left = background.width - MAX_BACKGROUND_WIDTH
    lowest_top = background.height - MAX_BACKGROUND_HEIGHT

    left = 0 if furthest_left <= 0 else np.random.randint(0, furthest_left)
    top = 0 if lowest_top <= 0 else np.random.randint(0, lowest_top)

    return copy.deepcopy(background).crop((left, top, left + MAX_BACKGROUND_WIDTH, top + MAX_BACKGROUND_HEIGHT))


def load_subjects(objects_dir):
    # Find subjects
    subjects = []
    label_names = set([])
    with os.scandir(objects_dir) as it:
        for entry in it:
            entry_root, ext = op.splitext(entry.name)
            if entry.is_file() and ext == '.png':
                label_name = entry_root
                for digit in string.digits:
                    label_name = label_name.replace(digit, '')
                label_names.add(label_name)
                subjects.append((entry.name, label_name))

    # Assign subject labels
    label_map = {label_name: label_id for label_id, label_name in enumerate(label_names)}

    # Create subjects
    subject_objects = []
    for (entry_name, label_name) in subjects:
        print('Processing subject: ', entry_name)
        subject_objects.append(Subject(objects_dir, entry_name, label_name, label_map[label_name]))

    return subject_objects, label_map


def generate_sample(subject, background, sampling_rates, draw_bounding_box=False):
    scaler = np.random.choice(sampling_rates)
    sample = Sample(subject, crop_background(background), scale=scaler, draw_bounding_box=draw_bounding_box)
    return sample


if __name__ == '__main__':
    set_name = 'all'

    scales = []
    backgrounds = load_backgrounds('backgrounds')
    subjects, label_map = load_subjects(objects_dir=set_name)

    sampling_rates = [0.25, 0.33, 0.41, 0.5, 0.59, 0.67, 0.75]

    num_subjects = len(subjects)
    num_backgrounds = len(backgrounds)
    print("num_subjects: ", num_subjects)
    print("num_backgrounds: ", num_backgrounds)
    print("RANDOM_SAMPLES_PER_OBJECT: ", RANDOM_SAMPLES_PER_OBJECT)
    num_samples = num_subjects*num_backgrounds*RANDOM_SAMPLES_PER_OBJECT
    print("num_samples: ", num_samples)

    #sample_names = []
    samples = np.zeros((num_samples, MAX_BACKGROUND_WIDTH, MAX_BACKGROUND_HEIGHT, NUM_CHANNELS)).astype('uint8')
    bounding_boxes = np.zeros((num_samples, 4)).astype('int32')
    labels = np.zeros((num_samples, 1)).astype('uint8')
    print("samples.shape: ", samples.shape, "\tsamples.dtype: ", samples.dtype)
    print("bounding_boxes.shape: ", bounding_boxes.shape, "\tlabels.dtype: ", bounding_boxes.dtype)

    sample_count = 0
    for subject in subjects:
        for idx, background in enumerate(backgrounds):
            for jdx in range(RANDOM_SAMPLES_PER_OBJECT):
                sample = generate_sample(subject, background, sampling_rates, draw_bounding_box=False)
                root, ext = op.splitext(sample.subject.image_name)
                new_image_name = root + '_' + str(idx) + '_' + str(jdx) + ext
                #sample_names.append(new_image_name)

                print('{}/{}: {}'.format(
                    sample_count+1,
                    num_samples,
                    new_image_name
                ))

                samples[sample_count] = np.array(sample.image.convert('RGB'))
                labels[sample_count] = subject.label_id
                bounding_boxes[sample_count] = np.array((
                    sample.bounding_box.top_left.x,
                    sample.bounding_box.top_left.y,
                    sample.bounding_box.bottom_right.x,
                    sample.bounding_box.bottom_right.y
                ))
                sample_count += 1

    '''
    # Validate data
    for idx, (sample_name, sample, label) in enumerate(zip(sample_names, samples, bounding_boxes)):
        sample_path = op.join('samples/', sample_name)
        print(sample_path, '\t', label, '\t', samples[idx].shape, '\t', samples[idx].dtype, '\t', np.min(samples[idx]), '\t', np.max(samples[idx]))
        img = Image.fromarray(samples[idx])
        img.save(sample_path)
    '''

    with open('{}_label_map.json'.format(set_name), 'w') as fp:
        fp.write(json.dumps(label_map))
        fp.close()

    np.savez('{}_samples.npz'.format(set_name), samples, labels, bounding_boxes)
