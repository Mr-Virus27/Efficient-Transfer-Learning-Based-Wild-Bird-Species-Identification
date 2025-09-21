from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input
from tensorflow.keras.optimizers import Adam

def build_model(input_shape, num_classes, lr=1e-4, freeze_base=True):
    base_model = VGG16(weights='imagenet', include_top=False,
                       input_tensor=Input(shape=input_shape))
    if freeze_base:
        for layer in base_model.layers:
            layer.trainable = False

    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)

    model.compile(optimizer=Adam(learning_rate=lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == '__main__':
    print('Import build_model from this module to construct the model.')
