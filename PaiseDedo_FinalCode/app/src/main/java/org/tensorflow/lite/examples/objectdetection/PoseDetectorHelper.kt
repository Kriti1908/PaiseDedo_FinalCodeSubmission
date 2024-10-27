package org.tensorflow.lite.examples.objectdetection

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.framework.image.MPImage
//import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.framework.image.BitmapImageBuilder


class PoseDetectorHelper(
    var threshold: Float = 0.5f,
    var numThreads: Int = 2,
    var currentDelegate: Int = DELEGATE_CPU,
    var maxResults: Int = 5, // Added maxResults property
    var currentModel: Int = 0, // Added currentModel property
    private val context: Context,
    private val poseDetectorListener: DetectorListener?
) {
    private var poseLandmarker: PoseLandmarker? = null

    init {
        setupPoseDetector()
    }

    fun clearPoseDetector() {
        poseLandmarker?.close()
        poseLandmarker = null
    }

    fun setupPoseDetector() {
        try {
            val baseOptionsBuilder = BaseOptions.builder()
                .setModelAssetPath("pose_landmarker_full.task")

            when (currentDelegate) {
                DELEGATE_CPU -> {
                    // Default is CPU
                }
                DELEGATE_GPU -> {
                    baseOptionsBuilder.setDelegate(Delegate.GPU)
                }
            }

            val baseOptions = baseOptionsBuilder.build()

            val options = PoseLandmarker.PoseLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setMinPoseDetectionConfidence(threshold)
                .setMinTrackingConfidence(threshold)
                .setRunningMode(RunningMode.IMAGE)
                .build()

            poseLandmarker = PoseLandmarker.createFromOptions(context, options)

        } catch (e: IllegalStateException) {
            poseDetectorListener?.onError(
                "Pose detector failed to initialize. See error logs for details"
            )
            Log.e("PoseDetectorHelper", "MediaPipe failed to load model with error: " + e.message)
        }
    }

    fun detectPose(image: Bitmap) {
        if (poseLandmarker == null) {
            setupPoseDetector()
        }

        try {
            // Convert Bitmap to MPImage
            val mpImage = BitmapImageBuilder(image).build()

            // Call detectAsync with MPImage and a timestamp
            poseLandmarker?.detectAsync(mpImage, System.currentTimeMillis())

        } catch (e: Exception) {
            poseDetectorListener?.onError(
                "Failed to detect pose. Error: " + e.message
            )
        }
    }


    interface DetectorListener {
        fun onResults(results: MutableList<PoseLandmarkerResult>?, inferenceTime: Long, imageHeight: Int, imageWidth: Int)
        fun onError(error: String)
    }

    companion object {
        const val DELEGATE_CPU = 0
        const val DELEGATE_GPU = 1
        const val DELEGATE_NNAPI = 2
    }
}
