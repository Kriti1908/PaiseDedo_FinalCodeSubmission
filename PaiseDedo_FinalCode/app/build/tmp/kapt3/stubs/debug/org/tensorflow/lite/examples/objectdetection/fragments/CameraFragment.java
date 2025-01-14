package org.tensorflow.lite.examples.objectdetection.fragments;

import java.lang.System;

@kotlin.Metadata(mv = {1, 6, 0}, k = 1, d1 = {"\u0000\u00c2\u0001\n\u0002\u0018\u0002\n\u0002\u0018\u0002\n\u0002\u0018\u0002\n\u0002\b\u0002\n\u0002\u0010\b\n\u0000\n\u0002\u0010\u000e\n\u0000\n\u0002\u0018\u0002\n\u0000\n\u0002\u0018\u0002\n\u0000\n\u0002\u0018\u0002\n\u0002\u0018\u0002\n\u0002\b\u0003\n\u0002\u0018\u0002\n\u0000\n\u0002\u0018\u0002\n\u0000\n\u0002\u0018\u0002\n\u0000\n\u0002\u0018\u0002\n\u0000\n\u0002\u0010\u0007\n\u0002\b\u0006\n\u0002\u0018\u0002\n\u0000\n\u0002\u0018\u0002\n\u0000\n\u0002\u0018\u0002\n\u0000\n\u0002\u0018\u0002\n\u0000\n\u0002\u0010\u0002\n\u0000\n\u0002\u0010!\n\u0002\u0018\u0002\n\u0000\n\u0002\u0018\u0002\n\u0002\b\u0002\n\u0002\u0018\u0002\n\u0000\n\u0002\u0018\u0002\n\u0002\b\u0002\n\u0002\u0018\u0002\n\u0002\b\t\n\u0002\u0018\u0002\n\u0000\n\u0002\u0018\u0002\n\u0000\n\u0002\u0018\u0002\n\u0000\n\u0002\u0018\u0002\n\u0000\n\u0002\u0018\u0002\n\u0002\b\u0006\n\u0002\u0010\t\n\u0002\b\u001a\u0018\u00002\u00020\u00012\u00020\u0002:\u0002ghB\u0005\u00a2\u0006\u0002\u0010\u0003J\b\u0010(\u001a\u00020)H\u0003J\u001a\u0010*\u001a\b\u0012\u0004\u0012\u00020,0+2\f\u0010-\u001a\b\u0012\u0004\u0012\u00020.0+J\u001e\u0010/\u001a\u00020,2\u0006\u00100\u001a\u0002012\f\u00102\u001a\b\u0012\u0004\u0012\u0002030+H\u0002J\u0010\u00104\u001a\u00020)2\u0006\u00105\u001a\u000206H\u0002J\u0010\u00107\u001a\u00020)2\u0006\u00105\u001a\u000206H\u0002J\b\u00108\u001a\u00020\u001aH\u0002J\b\u00109\u001a\u00020\u0005H\u0002J\b\u0010:\u001a\u00020\u001aH\u0002J\b\u0010;\u001a\u00020\u0007H\u0002J\b\u0010<\u001a\u00020\u0007H\u0002J\b\u0010=\u001a\u00020)H\u0002J\u0010\u0010>\u001a\u00020)2\u0006\u0010?\u001a\u00020@H\u0016J$\u0010A\u001a\u00020B2\u0006\u0010C\u001a\u00020D2\b\u0010E\u001a\u0004\u0018\u00010F2\b\u0010G\u001a\u0004\u0018\u00010HH\u0016J\b\u0010I\u001a\u00020)H\u0016J\u0010\u0010J\u001a\u00020)2\u0006\u0010K\u001a\u00020\u0007H\u0016J0\u0010L\u001a\u00020)2\u000e\u0010M\u001a\n\u0012\u0004\u0012\u00020.\u0018\u00010+2\u0006\u0010N\u001a\u00020O2\u0006\u0010P\u001a\u00020\u00052\u0006\u0010Q\u001a\u00020\u0005H\u0016J\b\u0010R\u001a\u00020)H\u0016J\u001a\u0010S\u001a\u00020)2\u0006\u0010T\u001a\u00020B2\b\u0010G\u001a\u0004\u0018\u00010HH\u0017J\b\u0010U\u001a\u00020)H\u0002J\u0010\u0010V\u001a\u00020)2\u0006\u0010W\u001a\u00020\u000bH\u0002J\b\u0010X\u001a\u00020)H\u0002J\b\u0010Y\u001a\u00020)H\u0002J\b\u0010Z\u001a\u00020)H\u0002J\b\u0010[\u001a\u00020)H\u0002J6\u0010\\\u001a\u00020)2\u0006\u0010W\u001a\u00020\u000b2\f\u0010]\u001a\b\u0012\u0004\u0012\u00020\u000e0\r2\u0006\u0010^\u001a\u00020\u001a2\u0006\u0010_\u001a\u00020\u00072\u0006\u0010`\u001a\u00020\u0005H\u0002J \u0010a\u001a\u00020)2\u0006\u0010b\u001a\u00020\u00052\u0006\u0010c\u001a\u00020\u001a2\u0006\u0010d\u001a\u00020\u001aH\u0002J\b\u0010e\u001a\u00020)H\u0002J\b\u0010f\u001a\u00020)H\u0002R\u000e\u0010\u0004\u001a\u00020\u0005X\u0082\u000e\u00a2\u0006\u0002\n\u0000R\u000e\u0010\u0006\u001a\u00020\u0007X\u0082D\u00a2\u0006\u0002\n\u0000R\u0010\u0010\b\u001a\u0004\u0018\u00010\tX\u0082\u000e\u00a2\u0006\u0002\n\u0000R\u000e\u0010\n\u001a\u00020\u000bX\u0082.\u00a2\u0006\u0002\n\u0000R\u0014\u0010\f\u001a\b\u0012\u0004\u0012\u00020\u000e0\rX\u0082\u0004\u00a2\u0006\u0002\n\u0000R\u000e\u0010\u000f\u001a\u00020\u000bX\u0082.\u00a2\u0006\u0002\n\u0000R\u0014\u0010\u0010\u001a\b\u0012\u0004\u0012\u00020\u000e0\rX\u0082\u0004\u00a2\u0006\u0002\n\u0000R\u000e\u0010\u0011\u001a\u00020\u0012X\u0082.\u00a2\u0006\u0002\n\u0000R\u0010\u0010\u0013\u001a\u0004\u0018\u00010\u0014X\u0082\u000e\u00a2\u0006\u0002\n\u0000R\u000e\u0010\u0015\u001a\u00020\u0016X\u0082.\u00a2\u0006\u0002\n\u0000R\u0010\u0010\u0017\u001a\u0004\u0018\u00010\u0018X\u0082\u000e\u00a2\u0006\u0002\n\u0000R\u000e\u0010\u0019\u001a\u00020\u001aX\u0082\u000e\u00a2\u0006\u0002\n\u0000R\u000e\u0010\u001b\u001a\u00020\u000bX\u0082.\u00a2\u0006\u0002\n\u0000R\u0014\u0010\u001c\u001a\b\u0012\u0004\u0012\u00020\u000e0\rX\u0082\u0004\u00a2\u0006\u0002\n\u0000R\u0014\u0010\u001d\u001a\u00020\t8BX\u0082\u0004\u00a2\u0006\u0006\u001a\u0004\b\u001e\u0010\u001fR\u0010\u0010 \u001a\u0004\u0018\u00010!X\u0082\u000e\u00a2\u0006\u0002\n\u0000R\u000e\u0010\"\u001a\u00020#X\u0082.\u00a2\u0006\u0002\n\u0000R\u000e\u0010$\u001a\u00020%X\u0082.\u00a2\u0006\u0002\n\u0000R\u0010\u0010&\u001a\u0004\u0018\u00010\'X\u0082\u000e\u00a2\u0006\u0002\n\u0000\u00a8\u0006i"}, d2 = {"Lorg/tensorflow/lite/examples/objectdetection/fragments/CameraFragment;", "Landroidx/fragment/app/Fragment;", "Lorg/tensorflow/lite/examples/objectdetection/PoseDetectorHelper$DetectorListener;", "()V", "CPU_Usage", "", "TAG", "", "_fragmentCameraBinding", "Lorg/tensorflow/lite/examples/objectdetection/databinding/FragmentCameraBinding;", "batteryConsumptionChart", "Lcom/github/mikephil/charting/charts/LineChart;", "batteryConsumptionEntries", "Ljava/util/ArrayList;", "Lcom/github/mikephil/charting/data/Entry;", "batteryLevelChart", "batteryLevelEntries", "bitmapBuffer", "Landroid/graphics/Bitmap;", "camera", "Landroidx/camera/core/Camera;", "cameraExecutor", "Ljava/util/concurrent/ExecutorService;", "cameraProvider", "Landroidx/camera/lifecycle/ProcessCameraProvider;", "chartXValue", "", "cpuUsageChart", "cpuUsageEntries", "fragmentCameraBinding", "getFragmentCameraBinding", "()Lorg/tensorflow/lite/examples/objectdetection/databinding/FragmentCameraBinding;", "imageAnalyzer", "Landroidx/camera/core/ImageAnalysis;", "metricLogger", "Lorg/tensorflow/lite/examples/objectdetection/MetricLogger;", "poseDetectorHelper", "Lorg/tensorflow/lite/examples/objectdetection/PoseDetectorHelper;", "preview", "Landroidx/camera/core/Preview;", "bindCameraUseCases", "", "convertPoseLandmarkerResultsToDetections", "", "Lorg/tensorflow/lite/examples/objectdetection/fragments/CameraFragment$Detection;", "poseLandmarkerResults", "Lcom/google/mediapipe/tasks/vision/poselandmarker/PoseLandmarkerResult;", "createDetectionInstance", "boundingBox", "Landroid/graphics/RectF;", "categories", "Lorg/tensorflow/lite/examples/objectdetection/fragments/CameraFragment$Category;", "detectObjects", "image", "Landroidx/camera/core/ImageProxy;", "detectPose", "getBatteryConsumption", "getBatteryLevel", "getCpuUsage", "getModelBasedOnCriteria", "getSelectedModel", "initBottomSheetControls", "onConfigurationChanged", "newConfig", "Landroid/content/res/Configuration;", "onCreateView", "Landroid/view/View;", "inflater", "Landroid/view/LayoutInflater;", "container", "Landroid/view/ViewGroup;", "savedInstanceState", "Landroid/os/Bundle;", "onDestroyView", "onError", "error", "onResults", "results", "inferenceTime", "", "imageHeight", "imageWidth", "onResume", "onViewCreated", "view", "setUpCamera", "setupChart", "chart", "setupCharts", "showGraphsSection", "showStatsSection", "startMetricUpdates", "updateChart", "entries", "newValue", "label", "color", "updateCharts", "batteryLevel", "cpuUsage", "batteryConsumption", "updateControlsUi", "updateSelectedModel", "Category", "Detection", "app_debug"})
public final class CameraFragment extends androidx.fragment.app.Fragment implements org.tensorflow.lite.examples.objectdetection.PoseDetectorHelper.DetectorListener {
    private int CPU_Usage = 0;
    private final java.lang.String TAG = "ObjectDetection";
    private org.tensorflow.lite.examples.objectdetection.databinding.FragmentCameraBinding _fragmentCameraBinding;
    private org.tensorflow.lite.examples.objectdetection.PoseDetectorHelper poseDetectorHelper;
    private android.graphics.Bitmap bitmapBuffer;
    private androidx.camera.core.Preview preview;
    private androidx.camera.core.ImageAnalysis imageAnalyzer;
    private androidx.camera.core.Camera camera;
    private androidx.camera.lifecycle.ProcessCameraProvider cameraProvider;
    private org.tensorflow.lite.examples.objectdetection.MetricLogger metricLogger;
    private com.github.mikephil.charting.charts.LineChart batteryLevelChart;
    private com.github.mikephil.charting.charts.LineChart cpuUsageChart;
    private com.github.mikephil.charting.charts.LineChart batteryConsumptionChart;
    private final java.util.ArrayList<com.github.mikephil.charting.data.Entry> batteryLevelEntries = null;
    private final java.util.ArrayList<com.github.mikephil.charting.data.Entry> cpuUsageEntries = null;
    private final java.util.ArrayList<com.github.mikephil.charting.data.Entry> batteryConsumptionEntries = null;
    private float chartXValue = 0.0F;
    
    /**
     * Blocking camera operations are performed using this executor
     */
    private java.util.concurrent.ExecutorService cameraExecutor;
    
    public CameraFragment() {
        super();
    }
    
    private final org.tensorflow.lite.examples.objectdetection.databinding.FragmentCameraBinding getFragmentCameraBinding() {
        return null;
    }
    
    private final void updateSelectedModel() {
    }
    
    private final java.lang.String getSelectedModel() {
        return null;
    }
    
    private final java.lang.String getModelBasedOnCriteria() {
        return null;
    }
    
    private final float getCpuUsage() {
        return 0.0F;
    }
    
    private final int getBatteryLevel() {
        return 0;
    }
    
    private final float getBatteryConsumption() {
        return 0.0F;
    }
    
    private final void updateCharts(int batteryLevel, float cpuUsage, float batteryConsumption) {
    }
    
    private final void updateChart(com.github.mikephil.charting.charts.LineChart chart, java.util.ArrayList<com.github.mikephil.charting.data.Entry> entries, float newValue, java.lang.String label, int color) {
    }
    
    @java.lang.Override
    public void onResume() {
    }
    
    @java.lang.Override
    public void onDestroyView() {
    }
    
    @org.jetbrains.annotations.NotNull
    @java.lang.Override
    public android.view.View onCreateView(@org.jetbrains.annotations.NotNull
    android.view.LayoutInflater inflater, @org.jetbrains.annotations.Nullable
    android.view.ViewGroup container, @org.jetbrains.annotations.Nullable
    android.os.Bundle savedInstanceState) {
        return null;
    }
    
    private final void setupChart(com.github.mikephil.charting.charts.LineChart chart) {
    }
    
    private final void setupCharts() {
    }
    
    @android.annotation.SuppressLint(value = {"MissingPermission"})
    @java.lang.Override
    public void onViewCreated(@org.jetbrains.annotations.NotNull
    android.view.View view, @org.jetbrains.annotations.Nullable
    android.os.Bundle savedInstanceState) {
    }
    
    private final void showStatsSection() {
    }
    
    private final void showGraphsSection() {
    }
    
    private final void startMetricUpdates() {
    }
    
    private final void initBottomSheetControls() {
    }
    
    private final void updateControlsUi() {
    }
    
    private final void setUpCamera() {
    }
    
    @android.annotation.SuppressLint(value = {"UnsafeOptInUsageError"})
    private final void bindCameraUseCases() {
    }
    
    private final void detectPose(androidx.camera.core.ImageProxy image) {
    }
    
    private final void detectObjects(androidx.camera.core.ImageProxy image) {
    }
    
    @java.lang.Override
    public void onConfigurationChanged(@org.jetbrains.annotations.NotNull
    android.content.res.Configuration newConfig) {
    }
    
    @org.jetbrains.annotations.NotNull
    public final java.util.List<org.tensorflow.lite.examples.objectdetection.fragments.CameraFragment.Detection> convertPoseLandmarkerResultsToDetections(@org.jetbrains.annotations.NotNull
    java.util.List<com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult> poseLandmarkerResults) {
        return null;
    }
    
    private final org.tensorflow.lite.examples.objectdetection.fragments.CameraFragment.Detection createDetectionInstance(android.graphics.RectF boundingBox, java.util.List<org.tensorflow.lite.examples.objectdetection.fragments.CameraFragment.Category> categories) {
        return null;
    }
    
    @java.lang.Override
    public void onResults(@org.jetbrains.annotations.Nullable
    java.util.List<com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult> results, long inferenceTime, int imageHeight, int imageWidth) {
    }
    
    @java.lang.Override
    public void onError(@org.jetbrains.annotations.NotNull
    java.lang.String error) {
    }
    
    @kotlin.Metadata(mv = {1, 6, 0}, k = 1, d1 = {"\u0000(\n\u0002\u0018\u0002\n\u0002\u0010\u0000\n\u0000\n\u0002\u0010\u000e\n\u0000\n\u0002\u0010\u0007\n\u0002\b\t\n\u0002\u0010\u000b\n\u0002\b\u0002\n\u0002\u0010\b\n\u0002\b\u0002\b\u0086\b\u0018\u00002\u00020\u0001B\u0015\u0012\u0006\u0010\u0002\u001a\u00020\u0003\u0012\u0006\u0010\u0004\u001a\u00020\u0005\u00a2\u0006\u0002\u0010\u0006J\t\u0010\u000b\u001a\u00020\u0003H\u00c6\u0003J\t\u0010\f\u001a\u00020\u0005H\u00c6\u0003J\u001d\u0010\r\u001a\u00020\u00002\b\b\u0002\u0010\u0002\u001a\u00020\u00032\b\b\u0002\u0010\u0004\u001a\u00020\u0005H\u00c6\u0001J\u0013\u0010\u000e\u001a\u00020\u000f2\b\u0010\u0010\u001a\u0004\u0018\u00010\u0001H\u00d6\u0003J\t\u0010\u0011\u001a\u00020\u0012H\u00d6\u0001J\t\u0010\u0013\u001a\u00020\u0003H\u00d6\u0001R\u0011\u0010\u0002\u001a\u00020\u0003\u00a2\u0006\b\n\u0000\u001a\u0004\b\u0007\u0010\bR\u0011\u0010\u0004\u001a\u00020\u0005\u00a2\u0006\b\n\u0000\u001a\u0004\b\t\u0010\n\u00a8\u0006\u0014"}, d2 = {"Lorg/tensorflow/lite/examples/objectdetection/fragments/CameraFragment$Detection;", "", "label", "", "score", "", "(Ljava/lang/String;F)V", "getLabel", "()Ljava/lang/String;", "getScore", "()F", "component1", "component2", "copy", "equals", "", "other", "hashCode", "", "toString", "app_debug"})
    public static final class Detection {
        @org.jetbrains.annotations.NotNull
        private final java.lang.String label = null;
        private final float score = 0.0F;
        
        @org.jetbrains.annotations.NotNull
        public final org.tensorflow.lite.examples.objectdetection.fragments.CameraFragment.Detection copy(@org.jetbrains.annotations.NotNull
        java.lang.String label, float score) {
            return null;
        }
        
        @java.lang.Override
        public boolean equals(@org.jetbrains.annotations.Nullable
        java.lang.Object other) {
            return false;
        }
        
        @java.lang.Override
        public int hashCode() {
            return 0;
        }
        
        @org.jetbrains.annotations.NotNull
        @java.lang.Override
        public java.lang.String toString() {
            return null;
        }
        
        public Detection(@org.jetbrains.annotations.NotNull
        java.lang.String label, float score) {
            super();
        }
        
        @org.jetbrains.annotations.NotNull
        public final java.lang.String component1() {
            return null;
        }
        
        @org.jetbrains.annotations.NotNull
        public final java.lang.String getLabel() {
            return null;
        }
        
        public final float component2() {
            return 0.0F;
        }
        
        public final float getScore() {
            return 0.0F;
        }
    }
    
    @kotlin.Metadata(mv = {1, 6, 0}, k = 1, d1 = {"\u0000(\n\u0002\u0018\u0002\n\u0002\u0010\u0000\n\u0000\n\u0002\u0010\u000e\n\u0000\n\u0002\u0010\u0007\n\u0002\b\t\n\u0002\u0010\u000b\n\u0002\b\u0002\n\u0002\u0010\b\n\u0002\b\u0002\b\u0086\b\u0018\u00002\u00020\u0001B\u0015\u0012\u0006\u0010\u0002\u001a\u00020\u0003\u0012\u0006\u0010\u0004\u001a\u00020\u0005\u00a2\u0006\u0002\u0010\u0006J\t\u0010\u000b\u001a\u00020\u0003H\u00c6\u0003J\t\u0010\f\u001a\u00020\u0005H\u00c6\u0003J\u001d\u0010\r\u001a\u00020\u00002\b\b\u0002\u0010\u0002\u001a\u00020\u00032\b\b\u0002\u0010\u0004\u001a\u00020\u0005H\u00c6\u0001J\u0013\u0010\u000e\u001a\u00020\u000f2\b\u0010\u0010\u001a\u0004\u0018\u00010\u0001H\u00d6\u0003J\t\u0010\u0011\u001a\u00020\u0012H\u00d6\u0001J\t\u0010\u0013\u001a\u00020\u0003H\u00d6\u0001R\u0011\u0010\u0002\u001a\u00020\u0003\u00a2\u0006\b\n\u0000\u001a\u0004\b\u0007\u0010\bR\u0011\u0010\u0004\u001a\u00020\u0005\u00a2\u0006\b\n\u0000\u001a\u0004\b\t\u0010\n\u00a8\u0006\u0014"}, d2 = {"Lorg/tensorflow/lite/examples/objectdetection/fragments/CameraFragment$Category;", "", "label", "", "score", "", "(Ljava/lang/String;F)V", "getLabel", "()Ljava/lang/String;", "getScore", "()F", "component1", "component2", "copy", "equals", "", "other", "hashCode", "", "toString", "app_debug"})
    public static final class Category {
        @org.jetbrains.annotations.NotNull
        private final java.lang.String label = null;
        private final float score = 0.0F;
        
        @org.jetbrains.annotations.NotNull
        public final org.tensorflow.lite.examples.objectdetection.fragments.CameraFragment.Category copy(@org.jetbrains.annotations.NotNull
        java.lang.String label, float score) {
            return null;
        }
        
        @java.lang.Override
        public boolean equals(@org.jetbrains.annotations.Nullable
        java.lang.Object other) {
            return false;
        }
        
        @java.lang.Override
        public int hashCode() {
            return 0;
        }
        
        @org.jetbrains.annotations.NotNull
        @java.lang.Override
        public java.lang.String toString() {
            return null;
        }
        
        public Category(@org.jetbrains.annotations.NotNull
        java.lang.String label, float score) {
            super();
        }
        
        @org.jetbrains.annotations.NotNull
        public final java.lang.String component1() {
            return null;
        }
        
        @org.jetbrains.annotations.NotNull
        public final java.lang.String getLabel() {
            return null;
        }
        
        public final float component2() {
            return 0.0F;
        }
        
        public final float getScore() {
            return 0.0F;
        }
    }
}