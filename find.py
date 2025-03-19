import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, LeakyReLU, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt

# ✅ Image size & grayscale input
IMG_SIZE = (128, 128, 1)  # Grayscale images
BATCH_SIZE = 32

# ✅ Load dataset with minimal augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,  
    rotation_range=10,   # Small rotation for robustness
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    shear_range=0.1,
    brightness_range=[0.8,1.2],
    horizontal_flip=True,
    validation_split=0.2
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    'Combined Dataset/train',
    target_size=IMG_SIZE[:2],  
    batch_size=BATCH_SIZE,
    color_mode='grayscale',  # ⚡ Convert to grayscale
    class_mode='categorical',
    subset='training'
)

val_data = val_datagen.flow_from_directory(
    'Combined Dataset/train',
    target_size=IMG_SIZE[:2],
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    class_mode='categorical',
    subset='validation'
)

# ✅ Residual Block (Enhances feature extraction)
# ✅ Fixed Residual Block
def residual_block(x, filters):
    shortcut = x  # Save original input
    
    # 🔥 Ensure shortcut has the correct number of channels
    if x.shape[-1] != filters:
        shortcut = Conv2D(filters, (1,1), padding='same', kernel_regularizer=l2(0.0001))(shortcut)
    
    x = Conv2D(filters, (3,3), padding='same', kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(filters, (3,3), padding='same', kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)

    x = Add()([shortcut, x])  # ✅ Now shapes match
    x = LeakyReLU()(x)
    
    return x


# ✅ CNN Model with Residual Blocks
input_layer = Input(shape=IMG_SIZE)

x = Conv2D(32, (3,3), kernel_regularizer=l2(0.0001), padding='same')(input_layer)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = MaxPooling2D(2,2)(x)

x = residual_block(x, 64)
x = MaxPooling2D(2,2)(x)

x = residual_block(x, 128)
x = residual_block(x, 128)
x = MaxPooling2D(2,2)(x)

x = residual_block(x, 256)
x = residual_block(x, 256)
x = MaxPooling2D(2,2)(x)

x = GlobalAveragePooling2D()(x)  # Reduce dimensions

x = Dense(256, kernel_regularizer=l2(0.0001))(x)
x = LeakyReLU()(x)
x = Dropout(0.4)(x)  

x = Dense(128, kernel_regularizer=l2(0.0001))(x)
x = LeakyReLU()(x)
x = Dropout(0.4)(x)

output_layer = Dense(4, activation='softmax')(x)  # Assuming 4 classes

model = Model(inputs=input_layer, outputs=output_layer)

optimizer=Adam(learning_rate=0.0001, clipnorm=1.0)

# ✅ Adaptive Learning Rate
model.compile(optimizer=optimizer,  
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ✅ Callbacks for better training
early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
checkpoint = ModelCheckpoint('best_model_brain_scan.h5', save_best_only=True, monitor='val_accuracy')

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=40,
    callbacks=[early_stopping, reduce_lr, checkpoint]
)

# ✅ Plot Training vs Validation Accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss')
plt.show()