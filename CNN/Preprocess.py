import numpy as np
import os
from PIL import Image
from tqdm import tqdm
# import splitfolders

# input = "/home/sami/Downloads/Sem 7/AI/Lab 13/kagglecatsanddogs_5340/PetImages"
# splitfolders.ratio(input, output=r"/home/sami/Downloads/Sem 7/AI/Lab 13", seed=100, ratio=(.7, 0.2,0.1))


def img_to_array(path):
    try:
        img = Image.open(path).convert("RGB")
    except:
        print(path)
        # os.remove(path)
    img = img.resize((64, 64))
    return np.array(img)

def main():
    train_path = "/home/sami/Downloads/Sem 7/AI/Lab 13/train"
    test_path = "/home/sami/Downloads/Sem 7/AI/Lab 13/test"
    val_path = "/home/sami/Downloads/Sem 7/AI/Lab 13/val"

    val_cats = os.listdir(os.path.join(val_path, "Cat"))
    val_dogs = os.listdir(os.path.join(val_path, "Dog"))
    train_cats = os.listdir(os.path.join(train_path, "Cat"))
    train_dogs = os.listdir(os.path.join(train_path, "Dog"))
    test_cats = os.listdir(os.path.join(test_path, "Cat"))
    test_dogs = os.listdir(os.path.join(test_path, "Dog"))

    train_data = np.zeros((len(train_cats) + len(train_dogs), 64, 64, 3))
    train_labels = []
    test_data = np.zeros((len(test_cats) + len(test_dogs), 64, 64, 3))
    test_labels = []
    val_data = np.zeros((len(val_cats) + len(val_dogs), 64, 64, 3))
    val_labels = []

    print("Loading data...")
    for i in tqdm(range(len(train_cats)+len(train_dogs))):
        if i < len(train_cats):
            train_data[i] = img_to_array(os.path.join(train_path, "Cat", train_cats[i]))
            train_labels.append(0)
        else:
            train_data[i] = img_to_array(os.path.join(train_path, "Dog", train_dogs[i-len(train_cats)]))
            train_labels.append(1)

    for i in tqdm(range(len(test_cats)+len(test_dogs))):
        if i < len(test_cats):
            test_data[i] = img_to_array(os.path.join(test_path, "Cat", test_cats[i]))
            test_labels.append(0)
        else:
            test_data[i] = img_to_array(os.path.join(test_path, "Dog", test_dogs[i-len(test_cats)]))
            test_labels.append(1)
    
    for i in tqdm(range(len(val_cats)+len(val_dogs))):
        if i < len(val_cats):
            val_data[i] = img_to_array(os.path.join(val_path, "Cat", val_cats[i]))
            val_labels.append(0)
        else:
            val_data[i] = img_to_array(os.path.join(val_path, "Dog", val_dogs[i-len(val_cats)]))
            val_labels.append(1)
    print("Data loaded successfully")
    print("Saving data...")
    np.save("train_data.npy", train_data)
    np.save("train_labels.npy", train_labels)
    np.save("test_data.npy", test_data)
    np.save("test_labels.npy", test_labels)
    np.save("val_data.npy", val_data)   
    np.save("val_labels.npy", val_labels)

if __name__ == "__main__":
    main()