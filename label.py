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
def draw_rectangle(event, x, y, flags, params):
    # if the left mouse button was clicked, record the starting coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        coords.append((x, y))
    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates
        coords.append((x, y))
        cv2.rectangle(image, coords[-2], coords[-1], (0, 255, 0), 2)
        cv2.imshow("image", image)


def save_to_yolo(categories, coords, outfile):
    with open(outfile, 'w') as f:
        for i in range(len(coords)//2):
            x1, y1 = coords[i*2]
            x2, y2 = coords[i*2+1]
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            width = abs(x2 - x1)
            height = abs(y2 - y1)
        
            center_x /= image_size[1]
            center_y /= image_size[0]
            width /= image_size[1]
            height /= image_size[0]
            # 切牌就是0， shuffle是i
            f.write(f"{categories[0]} {center_x:0.6f} {center_y:0.6f} {width:0.6f} {height:0.6f}\n")


    

root_dir = '.'
lableed = ['resized0750','resized0752','resized0753','resized0754', 'resized0755','resized0756','resized0757','Cutting_1','Cutting_1361','Cutting_1362']
data_dirs = ['Cutting_1363']
json_data = {}

for data_dir in data_dirs:
    if not os.path.exists(os.path.join(root_dir, 'Label', data_dir)):
        os.makedirs(os.path.join(root_dir, 'Label', data_dir))
    seq_image_dir = os.path.join(root_dir, 'CutImageData', data_dir)
    seq_label_file = os.path.join(root_dir, 'Label', data_dir.replace("Cutting", "Cutting_Label").replace("Shuffle", "Shuffle_Label").replace("resized", "Label") + '.json')
    images = sorted(os.listdir(seq_image_dir))
    with open(seq_label_file, 'r') as f:
        json_data[data_dir] = json.load(f)
    assert len(json_data[data_dir]) == len(images), print("#Label {} not equals #Image {} for seq {}".format(len(json_data[data_dir]), len(images), data_dir))
    for image_path in images:
        anno = json_data[data_dir][image_path]
        category = anno['Categories']
        image = cv2.imread(os.path.join(seq_image_dir, image_path))
        
        clone = image.copy()
        for i in range(0, len(coords), 2):
            cv2.rectangle(image, coords[i], coords[i+1], (0, 255, 0), 2)
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", draw_rectangle)

        while True:
            cv2.imshow("image", image)
            key = cv2.waitKey(1) & 0xFF
            # reset the drawing if 'r' is pressed
            if key == ord("r"):
                image = clone.copy()
                coords = []
            # break from the loop if 'c' is pressed (and save the coordinates to JSON)
            elif key == ord("c"):
                save_to_yolo(category, coords, os.path.join(root_dir, 'Label', data_dir, image_path.replace('.jpg', '.txt')))
                cv2.destroyAllWindows()
                break


        cv2.destroyAllWindows()
        print(image_path, coords)
