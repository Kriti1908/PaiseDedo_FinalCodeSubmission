package org.tensorflow.lite.examples.objectdetection

import android.graphics.RectF

data class Detection(
    val boundingBox: RectF, // The bounding box for the detected object
    val categories: List<Category> // List of categories for the detected object
)

