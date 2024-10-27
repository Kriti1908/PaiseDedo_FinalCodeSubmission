package org.tensorflow.lite.examples.objectdetection

data class Category(
    val label: String, // The label of the detected object
    val score: Float // The confidence score of the detection
)