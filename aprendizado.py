import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# Diretórios dos conjuntos de dados (barra normal evita o SyntaxWarning)
train_dir = './Treinamento'
validation_dir = './Validacao'

# Parâmetros do modelo
input_shape = (150, 150, 3)
num_classes = 4
batch_size = 32
epochs = 30  # o early stopping interrompe antes se necessário

# Modelo CNN com Input explícito e Dropout
model = Sequential([
    Input(shape=input_shape),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax'),
])

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# Treino com data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='categorical'
)

# Validação sem augmentation (apenas normalização)
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='categorical'
)

# Para o treino quando val_loss deixa de melhorar e restaura o melhor modelo
early_stop = EarlyStopping(
    monitor='val_loss', patience=4, restore_best_weights=True)

# Treina o modelo
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[early_stop],
)

# Avalia o modelo
loss, accuracy = model.evaluate(validation_generator)
print(f'Acurácia no conjunto de validação: {accuracy * 100:.2f}%')
print('Classes:', train_generator.class_indices)

# Salva o modelo treinado
model.save('modelo.keras')
