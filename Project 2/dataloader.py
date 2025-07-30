from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import cv2
import numpy as np

def custom_preprocessing(img):
    if np.random.rand() < 0.2:
        ksize = np.random.choice([3, 5])
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)

    if np.random.rand() < 0.2:
        noise = np.random.normal(0, 10, img.shape).astype(np.float32)
        img = img + noise
        img = np.clip(img, 0, 255)

    if np.random.rand() < 0.2:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), np.random.randint(50, 95)]
        result, encimg = cv2.imencode('.jpg', img.astype(np.uint8), encode_param)
        img = cv2.imdecode(encimg, 1)

    img = img.astype(np.float32)

    return img

def get_data_generators(data_dir="Dataset", img_size=(224, 224), batch_size=8):
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=15,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.7, 1.3],
        preprocessing_function=custom_preprocessing
    )

    val_test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(data_dir, "Train"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary'
    )

    val_gen = val_test_datagen.flow_from_directory(
        os.path.join(data_dir, "Validation"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )

    test_gen = val_test_datagen.flow_from_directory(
        os.path.join(data_dir, "Test"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )

    return train_gen, val_gen, test_gen
