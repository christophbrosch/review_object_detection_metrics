from object_detection_metrics.evaluators.coco_evaluator import get_coco_summary
from object_detection_metrics.evaluators.pascal_voc_evaluator import get_pascalvoc_metrics
from object_detection_metrics.utils.enumerators import BBFormat, BBType, CoordinatesType, FileFormat
from object_detection_metrics import read_pred_annotations
import object_detection_metrics.utils.converter as converter

def test_case_1():
    gts_dir = 'tests/test_case_1/gts'
    dets_dir = 'tests/test_case_1/dets'

    gts = converter.text2bb(gts_dir, BBType.GROUND_TRUTH)
    dets = converter.text2bb(dets_dir, BBType.DETECTED)
    get_pascalvoc_metrics(gts, dets)
    print(get_coco_summary(gts, dets))
    