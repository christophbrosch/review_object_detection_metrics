from object_detection_metrics.utils.enumerators import BBFormat, BBType, CoordinatesType, FileFormat
import object_detection_metrics.utils.converter as converter

def read_images():
    pass

def read_classes():
    pass

def read_gt_annotations(
    folder, 
    file_format: FileFormat, 
    bb_type = BBType.GROUND_TRUTH
):
    if file_format == FileFormat.YOLO:
        return converter.text2bb(folder, bb_type)

def read_pred_annotations(
    folder, 
    file_format: FileFormat, 
    bb_type = BBType.DETECTED, 
    bb_format = BBFormat.XYWH,
    type_coordinates = CoordinatesType.RELATIVE
):
    if file_format == FileFormat.YOLO:
        return converter.text2bb(folder, bb_type, bb_format, type_coordinates)