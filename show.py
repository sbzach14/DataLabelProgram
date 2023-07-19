import json
import cv2
import os
import json

suits = ['Spades', 'Hearts', 'Diamonds', 'Clubs']
numbers = list(range(1, 11)) + ['J', 'Q', 'K']
cat_to_num = {}
image_size = (540, 960)
i = 0
for suit in suits:
    for number in numbers:
        cat_to_num[i] = suit + str(number)
        i += 1

coords = []


root_dir = '.'
lableed = ['Shuffle_0022', ]
data_dirs = ['Shuffle_0024', 'Shuffle_0025', 'Shuffle_0027', 'Shuffle_0028', 'Shuffle_0030', 'Shuffle_0033', 'Shuffle_0034', 'Shuffle_0035', 'Shuffle_0037', 'Shuffle_0387', 'Shuffle_0389', 'Shuffle_0391', 'Shuffle_0392', 'Shuffle_0393']
json_data = {}

for data_dir in data_dirs:
    if not os.path.exists(os.path.join(root_dir, 'Label', data_dir)):
        os.makedirs(os.path.join(root_dir, 'Label', data_dir))
    seq_image_dir = os.path.join(root_dir, 'Image', data_dir)
    images = sorted(os.listdir(seq_image_dir))
    for image_path in images:
        image = cv2.imread(os.path.join(seq_image_dir, image_path))
        with open(os.path.join(root_dir, 'Label', data_dir, image_path.replace('.jpg', '.txt'))) as f:
            labels = f.readlines()
        for line in labels:
            category, x, y, w, h = line.split()
            x, y, w, h = float(x), float(y), float(w), float(h)
            category = int(category)
            x = x * image_size[1]
            y = y * image_size[0]
            w = w * image_size[1]
            h = h * image_size[0]
            bounding_box = [(int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2))]
            print(bounding_box, cat_to_num[category])
            cv2.rectangle(image, bounding_box[0], bounding_box[1], (0, 255, 0), 2)
            cv2.putText(image, cat_to_num[category], (bounding_box[0][0], bounding_box[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # cv2.putText(image, cat_to_num[category[i//2]], (coords[i][0], coords[i][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("image", image)
        cv2.waitKey(0)
            
        