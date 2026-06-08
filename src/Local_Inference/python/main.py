# File: main.py
# Written by: Angel Hernandez
# Description: Create an image classifier using Tiny ImageNet - http://cs231n.stanford.edu/tiny-imagenet-200.zip.
#              The program implements a CNN using TensorFlow. Model is then saved in ONNX format https://onnx.ai/
#              the purpose is to run inference locally using APIs available in Windows.

import os
import sys
import shutil
import zipfile
import subprocess
import urllib.request
from typing import Generic, TypeVar, Optional

T = TypeVar("T")

class ArgumentDefinition(Generic[T]):
    def __init__(self, name: str, default_value: T, read_arg: bool = True):
        self.name = name
        self.read_arg = read_arg
        self.value: Optional[T] = None
        self.default_value = default_value
        self.caster = type(default_value)

    def read(self):
        value = input(f"{self.name} (default {self.default_value}): ")
        if value.strip() == "":
            self.value = self.default_value
        else:
            try:
                self.value = self.caster(value)
            except Exception:
                raise ValueError(f"Invalid value for {self.name}: {value}")

class DatasetDownloadHelper:
    @staticmethod
    def prepare_tiny_imagenet():
        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        zip_path = "tiny-imagenet-200.zip"
        extract_dir = "tiny-imagenet-200"

        # Let's skip process if dataset has been downloaded previously
        if os.path.exists(extract_dir):
            print("Tiny ImageNet already exists. Skipping download.")
            return

        @staticmethod
        def show_download_progress(block_num, block_size, total_size):
            bar_len = 40
            downloaded = block_num * block_size
            percent = downloaded / total_size * 100
            filled = int(bar_len * percent / 100)
            bar = "█" * filled + "-" * (bar_len - filled)
            sys.stdout.write(f"\rDownloading: |{bar}| {percent:5.1f}%")
            sys.stdout.flush()

        print("\nDownloading Tiny ImageNet, please wait...")
        urllib.request.urlretrieve(url, zip_path, show_download_progress)
        print("\nDownload complete...")

        print("\nExtracting files in dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            total_files = len(file_list)
            for i, file in enumerate(file_list):
                zip_ref.extract(file, ".")
                percent = (i + 1) / total_files * 100
                bar_len = 40
                filled = int(bar_len * percent / 100)
                bar = "█" * filled + "-" * (bar_len - filled)
                sys.stdout.write(f"\rExtracting:  |{bar}| {percent:5.1f}%")
                sys.stdout.flush()
        print("\nDataset extraction complete...")
        print("\nReorganizing validation folder...")
        val_dir = os.path.join(extract_dir, "val")
        images_dir = os.path.join(val_dir, "images")
        annotations_file = os.path.join(val_dir, "val_annotations.txt")

        # Count lines for progress
        with open(annotations_file, "r") as f:
            lines = f.readlines()
        total = len(lines)

        for idx, line in enumerate(lines):
            parts = line.split("\t")
            img_file = parts[0]
            class_name = parts[1]
            class_dir = os.path.join(val_dir, class_name)

            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
            src = os.path.join(images_dir, img_file)
            dst = os.path.join(class_dir, img_file)

            if os.path.exists(src):
                shutil.move(src, dst)

            # Progress bar
            bar_len = 40
            percent = (idx + 1) / total * 100
            filled = int(bar_len * percent / 100)
            bar = "█" * filled + "-" * (bar_len - filled)
            sys.stdout.write(f"\rOrganizing:  |{bar}| {percent:5.1f}%")
            sys.stdout.flush()

        # Cleanup
        shutil.rmtree(images_dir)
        os.remove(annotations_file)
        print("\nValidation folder reorganized...")
        print("Tiny ImageNet is ready to use...")

class CnnConfig:
    def __init__(self, images_per_class=10, epochs=10, normalization_factor=255.0, batch_size=64,
                 validation_split=0.2, verbosity=1, training_rounds=2, epochs_per_training=5):
        self.epochs = epochs
        self.verbosity = verbosity
        self.batch_size = batch_size
        self.training_rounds = training_rounds
        self.validation_split = validation_split
        self.images_per_class = images_per_class
        self.epochs_per_training = epochs_per_training
        self.normalization_factor = normalization_factor

        arguments = [("epochs", ArgumentDefinition("Epochs", self.epochs)),
                     ("batch_size", ArgumentDefinition("Batch Size", self.batch_size)),
                     ("verbosity", ArgumentDefinition("Verbosity", self.verbosity)),
                     ("normalization_factor", ArgumentDefinition("Normalization Factor", self.normalization_factor)),
                     ("validation_split", ArgumentDefinition("Validation Split", self.validation_split)),
                     ("epochs_per_training", ArgumentDefinition("Epochs per Training Round", self.epochs_per_training)),
                     ("training_rounds", ArgumentDefinition("Training Rounds", self.training_rounds)),
                     ("images_per_class", ArgumentDefinition("Images per class", self.images_per_class))]
        for attr_name, arg in arguments:
            if arg.read_arg:
                 arg.read()
            value = arg.value if arg.value is not None else arg.default_value
            setattr(self, attr_name, value )

class DependencyChecker:
    @staticmethod
    def ensure_package(package_name):
        try:
            __import__(package_name)
        except ImportError:
            print(f"Installing missing package: {package_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"Package '{package_name}' was installed successfully.")

class TinyImageNetCnn:
    def __init__(self, cnn_config = None):
        if cnn_config is None:
            cnn_config = CnnConfig()
        self.cnn_config = cnn_config

        import numpy as np
        import tensorflow as tf
        import tf2onnx as tf2onnx

        self.__np = np
        self.__tf = tf
        self.__tf2onnx = tf2onnx
        self.model = None
        self.x_test = None
        self.y_test = None
        self.x_train = None
        self.y_train = None
        self.__class_names = []
        DatasetDownloadHelper.prepare_tiny_imagenet()
        self.__load_dataset()

    @staticmethod
    def suppress_warnings():
        os.environ['PYTHONHASHSEED'] = '0'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    def __load_dataset(self):
        img_size = (64, 64)
        print("\nLoading Tiny ImageNet...")
        data_dir = "tiny-imagenet-200"
        batch_size = self.cnn_config.batch_size

        # Training set
        self.train_ds = self.__tf.keras.preprocessing.image_dataset_from_directory(os.path.join(data_dir, "train"),
                           image_size=img_size, batch_size=batch_size, shuffle=True)

        # Validation set (must be reorganized beforehand)
        self.val_ds = self.__tf.keras.preprocessing.image_dataset_from_directory( os.path.join(data_dir, "val"),
                            image_size=img_size,  batch_size=batch_size, shuffle=True)

        self.__class_names = self.train_ds.class_names
        print(f"Loaded {len(self.__class_names)} classes.")

        # Normalization
        normalization = self.__tf.keras.layers.Rescaling(1.0 / self.cnn_config.normalization_factor)
        self.train_ds = self.train_ds.map(lambda x, y: (normalization(x), y))
        self.val_ds = self.val_ds.map(lambda x, y: (normalization(x), y))
        print("Dataset loaded and normalized.")

    def __build_model(self):
        print("Building Tiny ImageNet CNN model...\n")
        num_classes = len(self.__class_names)

        self.model = self.__tf.keras.Sequential([
            self.__tf.keras.Input(shape=(64, 64, 3)),
            self.__tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
            self.__tf.keras.layers.MaxPooling2D(),
            self.__tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
            self.__tf.keras.layers.MaxPooling2D(),
            self.__tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
            self.__tf.keras.layers.MaxPooling2D(),
            self.__tf.keras.layers.Flatten(),
            self.__tf.keras.layers.Dense(256, activation='relu'),
            self.__tf.keras.layers.Dropout(0.3),
            self.__tf.keras.layers.Dense(num_classes, activation='softmax')
        ])

        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print("\nModel built successfully.")
        return self.model

    def __train_model(self, model, epochs=10):
        print(f"Training CNN for {epochs} epochs...\n")
        self.history = model.fit(self.train_ds, validation_data=self.val_ds, epochs=epochs,
                                 verbose=self.cnn_config.verbosity)
        print("Training complete.")
        return self.history

    def __evaluate_model(self, model):
        print("Evaluating model on validation dataset...")
        loss, accuracy = model.evaluate(self.val_ds, verbose=0)
        print(f"Validation Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")
        return loss, accuracy

    def __predict_sample(self, model, index=0):
        print("Predicting a sample image from validation set...")

        for batch_images, batch_labels in self.val_ds.take(1):
            img = batch_images[index:index + 1]
            label = batch_labels[index].numpy()
            prediction = model.predict(img, verbose=0)
            predicted_class = self.__np.argmax(prediction)
            print(f"Predicted: {self.__class_names[predicted_class]}, Actual: {self.__class_names[label]}")

    def __repeat_training(self, model, rounds=3, epochs=5):
        print(f"Repeating training {rounds} times to reduce loss...")
        for r in range(rounds):
            print(f"\nTraining Round: {r + 1}/{rounds}")
            self.__train_model(model, epochs=epochs)

    def train_and_classify(self):
        model = self.__build_model()
        # Initial training
        history = self.__train_model(model, epochs=self.cnn_config.epochs)
        # Repeat training to reduce loss
        self.__repeat_training(model, rounds=self.cnn_config.training_rounds, epochs=self.cnn_config.epochs_per_training)
        # Evaluate
        self.__evaluate_model(model)
        # Predict a sample
        self.__predict_sample(model, index=0)
        self.__export_to_onnx()

    def __export_to_onnx(self, output_path="tiny_imagenet_classifier.onnx"):
        print("\nExporting model to ONNX format...")

        # Extract the actual input tensor name (e.g., "input_1:0")
        raw_name = self.model.inputs[0].name
        input_name = raw_name.split(":")[0]  # remove ":0"

        spec = (self.__tf.TensorSpec((None, 64, 64, 3), self.__tf.float32, name=input_name),)

        model_proto, _ = self.__tf2onnx.convert.from_keras(self.model, input_signature=spec, opset=17,
                                                          output_path=output_path)
        print(f"ONNX model saved to: {output_path}")

class TestCaseRunner:
    @staticmethod
    def run_scenario():
        cnn = TinyImageNetCnn()
        cnn.train_and_classify()

def clear_screen():
    command = 'cls' if os.name == 'nt' else 'clear'
    os.system(command)

def main():
    try:
        TinyImageNetCnn.suppress_warnings()
        dependencies = ['numpy', 'pandas', 'tensorflow','tf2onnx']
        for d in dependencies: DependencyChecker.ensure_package(d)
        clear_screen()
        print('*** Image Classifier (CNN) using Tiny ImageNet | Model saved in ONNX format  ***\n')
        TestCaseRunner.run_scenario()
    except Exception as e:
        print(e)

if __name__ == '__main__':
    main()