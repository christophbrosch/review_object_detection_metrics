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
    results_pascal_voc = get_pascalvoc_metrics(gts, dets)
    results_coco = get_coco_summary(gts, dets)

def test_case_2():
    gts_dir = 'tests/test_coco_eval/gts'
    dets_dir = 'tests/test_coco_eval/dets'

    gts = converter.coco2bb(gts_dir, BBType.GROUND_TRUTH)
    dets = converter.coco2bb(dets_dir, BBType.DETECTED)
    results_pascal_voc = get_pascalvoc_metrics(gts, dets)
    results_coco = get_coco_summary(gts, dets)

    print(results_coco)