import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define os diretórios dos conjuntos de dados para cada classe
train_dir = '.\Treinamento'
validation_dir = '.\Validacao'

# Define os parâmetros do modelo
input_shape = (150, 150, 3)
num_classes = 4
batch_size = 32
epochs = 10

# Cria um modelo de rede neural convolucional simples
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compila o modelo
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Configura a geração de dados de treinamento e validação
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='categorical'
)

# Treina o modelo
history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator)

# Avalia o modelo
accuracy = model.evaluate(validation_generator)
print(f'Acurácia no conjunto de validação: {accuracy[1] * 100:.2f}%')