import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.applications import VGG16
from sklearn.model_selection import train_test_split

# Đường dẫn dữ liệu
train_dir = 'C:/Users/admin/Downloads/XLABai1/baitap16.11/data/Train'
validation_dir = 'C:/Users/admin/Downloads/XLABai1/baitap16.11/data/Validation'

# Tạo bộ dữ liệu với ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Load dữ liệu
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

# Mô hình ANN
ann_model = Sequential([
    Flatten(input_shape=(128, 128, 3)),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile mô hình
ann_model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

ann_model.summary()

# Mô hình CNN
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile mô hình
cnn_model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

cnn_model.summary()

# Sử dụng VGG16 để trích xuất đặc trưng
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Đóng băng các lớp trong VGG16
for layer in base_model.layers:
    layer.trainable = False

# Mô hình R-CNN
rcnn_model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile mô hình
rcnn_model.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])

rcnn_model.summary()

# Huấn luyện ANN
print("Training ANN Model...")
history_ann = ann_model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# Huấn luyện CNN
print("Training CNN Model...")
history_cnn = cnn_model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# Huấn luyện R-CNN
print("Training R-CNN Model...")
history_rcnn = rcnn_model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# Đánh giá mô hình
print("Evaluating ANN Model...")
ann_loss, ann_acc = ann_model.evaluate(validation_generator)
print(f"ANN Accuracy: {ann_acc*100:.2f}%")

print("Evaluating CNN Model...")
cnn_loss, cnn_acc = cnn_model.evaluate(validation_generator)
print(f"CNN Accuracy: {cnn_acc*100:.2f}%")

print("Evaluating R-CNN Model...")
rcnn_loss, rcnn_acc = rcnn_model.evaluate(validation_generator)
print(f"R-CNN Accuracy: {rcnn_acc*100:.2f}%")

# Hàm dự đoán một ảnh mới
def predict_image(model, img_path):
    img = load_img(img_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    if prediction[0] > 0.5:
        print("Prediction: Chó")
    else:
        print("Prediction: Mèo")

# Thử dự đoán
test_image_path = 'C:/Users/admin/Downloads/XLABai1/baitap16.11/data/Validation/Mèo/meo01.jpg'  # Cập nhật đường dẫn
print("Prediction using CNN Model:")
predict_image(cnn_model, test_image_path)
