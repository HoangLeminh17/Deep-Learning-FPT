from dataloader import get_data_generators
from utils import get_callbacks
from model_build import build_resnet50
import tensorflow as tf
from tensorflow.keras.optimizers import Adam


img_size = (224, 224)
batch_size = 8
epochs = 50
data_dir = 'Dataset'
model_path = 'model.h5'

train_gen, val_gen, _ = get_data_generators(data_dir=data_dir,
                                            img_size=img_size,
                                            batch_size=batch_size)

model, base_model = build_resnet50(input_shape=img_size + (3,))

print("\n==============================")
print("PHASE 1a: Fine-tune block 5")
print("==============================\n")

for layer in base_model.layers:
    layer.trainable = False

for layer in base_model.layers:
    if 'conv5_block' in layer.name:
        layer.trainable = True

model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_gen, validation_data=val_gen, epochs=5, callbacks=get_callbacks(model_path=model_path))

print("\n==============================")
print("PHASE 1b: Fine-tune block 4 & 5")
print("==============================\n")

for layer in base_model.layers:
    layer.trainable = False

for layer in base_model.layers:
    if 'conv4_block' in layer.name or 'conv5_block' in layer.name:
        layer.trainable = True

model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_gen, validation_data=val_gen, epochs=10, initial_epoch=5, callbacks=get_callbacks(model_path=model_path))

print("\n==============================")
print("PHASE 2: Fine-tune FULL backbone")
print("==============================\n")

for layer in base_model.layers:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_gen, validation_data=val_gen, epochs=40, initial_epoch=10, callbacks=get_callbacks(model_path=model_path))

