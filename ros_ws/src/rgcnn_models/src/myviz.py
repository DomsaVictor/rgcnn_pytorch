#!/home/victor/anaconda3/envs/thesis_env/bin/python3

import roslib

import sys

from python_qt_binding.QtGui import *
from python_qt_binding.QtCore import *
from python_qt_binding.QtWidgets import *


def slider_function(value):
    print("Slider Function" + str(value))

class MySlider(QWidget):
    """
    You can add sliders with the 'addSlider' function
    """
    def __init__(self):
        self.app = QApplication( sys.argv )

        QWidget.__init__(self)

        self.layout = QVBoxLayout(self)

        # self.addSlider("X", 1, 1000, self.printValue)
        # self.addSlider("Wasd", 1, 30, self.printValue)

    
    def addSlider(self, label_text, min=0, max=1000, function=None):
        """
        Adds a slider to the window.
        Parameters:
            label_text: text to display at the left of the slider. Currently is capped at 8 characters for alignment reasons
            min:        minimum value of the slider
            max:        maximum value of the slider
            function:   function to be executed when slider value changes. Can be defined outside the class, just as a normal function
                        It must have one parameter which represents the changed value (ex: def sliderChanged(newSliderValue): ...)
        """

        assert function is not None

        new_slider = QSlider(Qt.Horizontal)
        new_slider.setTracking(True)
        new_slider.setMinimum(min)
        new_slider.setMaximum(max)
        new_slider.valueChanged.connect(function)

        if len(label_text) < 8:
            label_text = label_text + "\t"
        else:
            label_text = label_text[0:8]
        new_label = QLabel(label_text, alignment=Qt.AlignLeft)

        new_H_layout = QHBoxLayout()
        new_H_layout.addWidget(new_label)
        new_H_layout.addWidget(new_slider)

        self.layout.addLayout(new_H_layout)

    def printValue(self, valueChanged):
        """
        Just a model function
        """
        print(valueChanged)

    def start(self):
        self.show()
        self.app.exec_()


if __name__ == '__main__':
    app = QApplication( sys.argv )

    myviz = MySlider()
    myviz.addSlider("text", function=slider_function)
    myviz.start()
    # myviz.show()

    # app.exec_()