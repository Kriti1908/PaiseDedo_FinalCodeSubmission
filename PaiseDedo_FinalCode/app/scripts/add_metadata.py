model_path = "../src/main/assets/mediapipe_pose-mediapipeposedetector.tflite"
from tflite_support import flatbuffers
from tflite_support import metadata as _metadata
from tflite_support import metadata_schema_py_generated as _metadata_fb
import os
import tensorflow as tf

def get_model_details(model_path):
    """
    Get actual input/output tensor details from the TFLite model
    """
    # Load the TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("Input Details:", input_details)
    print("Output Details:", output_details)
    
    return input_details, output_details

def create_pose_model_metadata(model_path):
    """
    Create metadata for MediaPipe Pose detection model using actual model info
    """
    # Get actual model details
    input_details, output_details = get_model_details(model_path)
    
    # Create metadata for the model
    model_meta = _metadata_fb.ModelMetadataT()
    model_meta.name = "MediaPipe Pose Detector"
    model_meta.description = "Detect human pose landmarks in images/video frames."
    model_meta.version = "v1"
    model_meta.author = "MediaPipe"
    model_meta.license = "Apache License. Version 2.0"

    # Create input tensor info based on actual model input
    input_metas = []
    for input_detail in input_details:
        input_meta = _metadata_fb.TensorMetadataT()
        input_meta.name = input_detail['name']
        # input_meta.description = f"Input tensor of shape {input_detail['shape']}"
        # input_meta.content = _metadata_fb.ContentT()
        # input_meta.content.content_properties = _metadata_fb.ImagePropertiesT()
        # input_meta.content.contentPropertiesType = (
        #     _metadata_fb.ContentProperties.ImageProperties)
        
        # # Get actual input stats
        # input_stats = _metadata_fb.StatsT()
        # input_stats.max = [float(input_detail.get('quantization_parameters', {}).get('max', [1.0])[0])]
        # input_stats.min = [float(input_detail.get('quantization_parameters', {}).get('min', [0.0])[0])]
        # input_meta.stats = input_stats
        
        input_metas.append(input_meta)

    # Create output tensor info based on actual model outputs
    output_metas = []
    for output_detail in output_details:
        output_meta = _metadata_fb.TensorMetadataT()
        output_meta.name = output_detail['name']
        # output_meta.description = f"Output tensor of shape {output_detail['shape']}"
        # output_meta.content = _metadata_fb.ContentT()
        # output_meta.content.content_properties = _metadata_fb.FeaturePropertiesT()
        # output_meta.content.contentPropertiesType = (
        #     _metadata_fb.ContentProperties.FeatureProperties)
        
        # # Get actual output stats
        # output_stats = _metadata_fb.StatsT()
        # output_stats.max = [float(output_detail.get('quantization_parameters', {}).get('max', [1.0])[0])]
        # output_stats.min = [float(output_detail.get('quantization_parameters', {}).get('min', [-1.0])[0])]
        # output_meta.stats = output_stats
        
        output_metas.append(output_meta)

    # Create subgraph info
    subgraph = _metadata_fb.SubGraphMetadataT()
    subgraph.inputTensorMetadata = input_metas
    subgraph.outputTensorMetadata = output_metas
    model_meta.subgraphMetadata = [subgraph]

    # Create metadata buffer
    b = flatbuffers.Builder(0)
    b.Finish(
        model_meta.Pack(b),
        _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
    metadata_buf = b.Output()

    # Populate model with metadata
    populator = _metadata.MetadataPopulator.with_model_file(model_path)
    populator.load_metadata_buffer(metadata_buf)
    populator.populate()
    
    return populator

def verify_metadata(model_path):
    """
    Verify and display metadata from the model
    """
    displayer = _metadata.MetadataDisplayer.with_model_file(model_path)
    metadata_json = displayer.get_metadata_json()
    print("\nModel Metadata:")
    print(metadata_json)
    
    return metadata_json

try:
    # First, print the actual model details
    print("Fetching model details...")
    input_details, output_details = get_model_details(model_path)
    
    print("\nApplying metadata based on actual model information...")
    populator = create_pose_model_metadata(model_path)
    
    print("\nVerifying applied metadata...")
    metadata = verify_metadata(model_path)
    
except Exception as e:
    print(f"Error occurred: {str(e)}")
    print("Stack trace:")
    import traceback
    traceback.print_exc()