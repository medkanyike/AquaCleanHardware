import tflite_runtime.interpreter as tflite
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import PIL
import cv2
import numpy as np
import argparse
import sys
import time
import RPi.GPIO as GPIO
import datetime

# Hardware imports and setup
from time import sleep


class Motor:
    def __init__(self, ena, in1, in2):
        self.ena = ena
        self.in1 = in1
        self.in2 = in2
        self.temp = 1
        GPIO.setup(self.in1, GPIO.OUT)
        GPIO.setup(self.in2, GPIO.OUT)
        GPIO.setup(self.ena, GPIO.OUT)
        GPIO.output(self.in1, GPIO.LOW)
        GPIO.output(self.in2, GPIO.LOW)
        self.p = GPIO.PWM(self.ena, 1000)
        self.p.start(25)

    def run(self):
        print("run")
        if self.temp == 1:
            GPIO.output(self.in1, GPIO.HIGH)
            GPIO.output(self.in2, GPIO.LOW)
            print("forward")
        else:
            GPIO.output(self.in1, GPIO.LOW)
            GPIO.output(self.in2, GPIO.HIGH)
            print("backward")

    def stop(self):
        print("stop")
        GPIO.output(self.in1, GPIO.LOW)
        GPIO.output(self.in2, GPIO.LOW)

    def forward(self):
        print("forward")
        GPIO.output(self.in1, GPIO.HIGH)
        GPIO.output(self.in2, GPIO.LOW)
        self.temp = 1

    def backward(self):
        print("backward")
        GPIO.output(self.in1, GPIO.LOW)
        GPIO.output(self.in2, GPIO.HIGH)
        self.temp = 0

    def slow(self):
        print("slow")
        self.p.ChangeDutyCycle(25)

    def medium(self):
        print("medium")
        self.p.ChangeDutyCycle(50)

    def fast(self):
        print("high")
        self.p.ChangeDutyCycle(75)

    @staticmethod
    def exit():
        GPIO.cleanup()


class UltrasonicSensor:
    def __init__(self, trigger_pin, echo_pin):
        self.trigger_pin = trigger_pin
        self.echo_pin = echo_pin
        # Setup GPIO mode
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.trigger_pin, GPIO.OUT)
        GPIO.setup(self.echo_pin, GPIO.IN)
        # Ensure trigger is low initially
        GPIO.output(self.trigger_pin, False)
        time.sleep(2)

    def measure_distance(self):
        try:
            # Send 10us pulse to trigger
            GPIO.output(self.trigger_pin, True)
            time.sleep(0.00001)  # 10 microseconds
            GPIO.output(self.trigger_pin, False)

            # Measure the time of the echo
            while GPIO.input(self.echo_pin) == 0:
                pulse_start = time.time()

            while GPIO.input(self.echo_pin) == 1:
                pulse_end = time.time()

            pulse_duration = pulse_end - pulse_start

            # Calculate the distance
            distance = (
                pulse_duration * 17150
            )  # Speed of sound at 34300 cm/s (17150 cm/s each way)
            distance = round(distance, 2)

            return distance
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def cleanup(self):
        GPIO.cleanup()


# Define paths
TF_MODEL_FILE_PATH = "model.tflite"
# GPIO.setmode(GPIO.BCM)

ON = True
OFF = False


# Image processing functions
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    # Convert to UINT8 and scale to [0, 255] range (if necessary)
    if image.dtype == np.float32:
        image = (image * 255.0).astype(np.uint8)  # Scale and convert
    # Add batch dimension
    return np.expand_dims(image, axis=0)


def detect_bottles(image):
    interpreter = tflite.Interpreter(model_path=TF_MODEL_FILE_PATH)
    interpreter.allocate_tensors()

    # Load the image
    preprocessed_image = preprocess_image(image)

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Run inference
    interpreter.set_tensor(input_details[0]["index"], preprocessed_image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])

    # Process the output
    if output_data.ndim == 1:  # Assuming single output value (e.g., regression)
        print(f"Predicted value: {output_data[0]}")
    elif output_data.ndim == 2:
        # Assuming classification output (probabilities)
        predictions = output_data
        predicted_class = np.argmax(predictions)

        with open(class_names_file, "r") as f:
            # Read all lines from the file
            class_names = f.readlines()
            # Remove any trailing newline characters from each class name
            class_names = [line.strip() for line in class_names]

        return class_names[predicted_class]
    else:
        print("There is a problem connecting the camera")


def capture_and_process_image():
    # Capture the image
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 680)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    ret, frame = cap.read()
    cap.release()  # Release the camera resource

    return detect_bottles(frame)


def isBottle(text):
    # Convert text to lowercase for case-insensitive check
    lower_text = text.lower()
    # Check if "bottle" is present as a whole word or part of a compound word
    return ("bottle" in lower_text) or (
        " " in lower_text and "bottle" in lower_text.split()
    )


def main():
    GPIO.setmode(GPIO.BCM)
    motor1 = Motor(25, 24, 23)
    motor2 = Motor(8, 7, 1)
    sensor = UltrasonicSensor(trigger_pin=17, echo_pin=27)

    try:
        while True:
            capturedObject = capture_and_process_image()
            distance = sensor.measure_distance()
            if distance:
                print(f"Distance: {distance} cm")
            if isBottle(capturedObject):
                motor1.run()
                motor2.run()
                sleep(2)
                motor1.forward()
                motor2.forward()
                motor1.fast()
                motor2.fast()
            else:
                motor2.backward()
                motor1.backward()

            time.sleep(2)
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        motor1.stop()
        motor2.stop()
        sensor.cleanup()


if __name__ == "__main__":
    main()
