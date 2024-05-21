import tensorflow as tf
from load_mnist import load_data


num_classes = 10

def model_function(trial):
    num_filters = trial.suggest_categorical('num_filters', [32, 64, 128])
    filter_size = trial.suggest_categorical('filter_size', [3, 5])
    stride = trial.suggest_categorical('stride', [1, 2])
    padding = trial.suggest_categorical('padding', ['valid', 'same'])
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5, step=0.1)
    pool_size = trial.suggest_categorical('pool_size', [2, 3])

    training_data, validation_data, test_data = load_data()
    training_data = list(training_data)
    test_data = list(test_data)

    X_train, X_test, y_train, y_test = training_data[0], test_data[0], training_data[1], test_data[1]

    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(filter_size, filter_size), activation='relu', 
                                     strides=(stride, stride), padding=padding, input_shape=(28, 28, 1)))
    current_shape = (28, 28)
    if padding == 'valid':
        current_shape = ((current_shape[0] - filter_size + stride) // stride, 
                         (current_shape[1] - filter_size + stride) // stride)
    else:
        current_shape = ((current_shape[0] + stride - 1) // stride, 
                         (current_shape[1] + stride - 1) // stride)

    if all(dim >= pool_size for dim in current_shape):
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(pool_size, pool_size)))
        current_shape = (current_shape[0] // pool_size, current_shape[1] // pool_size)

    model.add(tf.keras.layers.Conv2D(filters=num_filters * 2, kernel_size=(filter_size, filter_size), activation='relu', 
                                     strides=(stride, stride), padding=padding))
    
    if padding == 'valid':
        current_shape = ((current_shape[0] - filter_size + stride) // stride, 
                         (current_shape[1] - filter_size + stride) // stride)
    else:
        current_shape = ((current_shape[0] + stride - 1) // stride, 
                         (current_shape[1] + stride - 1) // stride)

    if all(dim >= pool_size for dim in current_shape):
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(pool_size, pool_size)))
        current_shape = (current_shape[0] // pool_size, current_shape[1] // pool_size)

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))
    model.add(tf.keras.layers.Dense(units=num_classes, activation='softmax'))



    # Compile the model
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy']
    )


    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    return accuracy


def other_params_model(trial):
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    epochs = trial.suggest_int('epochs', 5, 20)
    batch_size = trial.suggest_int('batch_size', 32, 128)
    num_filters = trial.suggest_categorical('num_filters', [32, 64, 128])
    dense_units = trial.suggest_int('dense_units', 64, 256)

    training_data, validation_data, test_data = load_data()
    training_data = list(training_data)
    test_data = list(test_data)

    X_train, X_test, y_train, y_test = training_data[0], test_data[0], training_data[1], test_data[1]

    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(filters=num_filters * 2, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),  # Flatten the data for dense layers
        tf.keras.layers.Dense(units=dense_units, activation='relu'),
        tf.keras.layers.Dense(units=num_classes, activation='softmax')  # Output layer with softmax for probabilities
    ])

    # Compile the model
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=['accuracy']
    )

    # Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    return accuracy
