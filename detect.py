# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # datasets.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    # 如果nosave为False（意味着允许保存图像）且source不以.txt结尾，那么将设置save_img为True
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    # 检查source是否为一个文件。它获取source的后缀并检查它是否在图片格式和视频格式列表中
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        # 如果source是URL且同时是一个文件，那么将调用check_file函数来下载文件
        source = check_file(source)  # download

    # Directories
    # 通过increment_path函数创建一个新的保存目录路径。Path(project)表示项目的路径，name是文件夹的名称。
    # increment_path函数的作用是在路径后面添加一个索引以确保目录的唯一性。exist_ok参数表示如果目录已存在，是否允许覆盖
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    # 这行代码根据是否保存文本文件（labels），创建目录。如果save_txt为True，
    # 那么在save_dir路径下创建一个名为'labels'的子目录，用于保存文本文件。如果save_txt为False，就直接在save_dir路径下创建目录
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    # 这行代码创建一个目标检测模型，基于DetectMultiBackend类。它使用了一些参数，如weights（模型权重路径）、device（计算设备）、dnn、data和fp16（是否使用半精度浮点数运算）
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    # 这行代码从model对象中获取了stride（步长）、names（类别名称）和pt（模型权重）的值。这些值可能在后续的操作中使用
    stride, names, pt = model.stride, model.names, model.pt
    # 这行代码调用check_img_size函数，用于检查输入的图像尺寸（imgsz）。它可能会调整图像尺寸，以确保它适合模型的要求，同时考虑到stride的影响
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        # 用于检查是否显示图像。参数warn=True表示在没有GUI界面时会显示警告信息
        view_img = check_imshow(warn=True)
        # 将创建一个LoadStreams数据集，用于加载摄像头流。img_size表示图像尺寸，stride表示步长，auto和vid_stride可能会被用来设置不同的参数
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        # 这里将批量大小（bs）设置为数据集的长度。这可能是为了在摄像头模式下，每个批次都包含一帧图像
        bs = len(dataset)
    elif screenshot:
        # 如果使用屏幕截图，将创建一个LoadScreenshots数据集，用于加载屏幕截图
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        # 加载图像数据
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    # 创建了两个列表，vid_path和vid_writer，长度为批量大小（bs）。这些列表可能会在后续用于存储视频路径和写入视频的对象
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    # 这行代码用于预热模型，以便更好地运行推理。它设置了预热的图像尺寸
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    # 这里初始化了一些变量，包括seen（已处理的图像数量）、windows（窗口信息列表）和dt（计时器的元组）
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    # 将输入图像准备好，然后执行目标检测推理。在循环中，它遍历数据集中的每个图像，将其准备好，并使用模型进行推理。
    # 这是一个循环，用于遍历数据集中的每个图像。在每次迭代中，会提取path（图像路径）、im（图像数组）、im0s（原始图像数组，未调整大小）、vid_cap（视频捕获对象）和s（图像的步长）
    for path, im, im0s, vid_cap, s in dataset:
        # 使用计时器来测量推理时间
        with dt[0]:
            # 将im从NumPy数组转换为PyTorch张量，并将其移动到模型所在的设备（CPU或GPU）
            im = torch.from_numpy(im).to(model.device)
            # 如果模型使用半精度浮点数计算（fp16=True），则将图像张量转换为半精度浮点数（float16）。否则，将其转换为单精度浮点数（float32）
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            # 将图像的像素值从0-255的范围映射到0.0-1.0的范围
            im /= 255  # 0 - 255 to 0.0 - 1.0
            # 这行代码检查图像张量是否缺少批量维度（batch dimension），如果是，则在维度0处添加一个批量维度
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        # 执行目标检测推理并进行后处理，包括可视化和非最大抑制。在每次迭代中，它将图像传递给模型进行推理，然后对预测结果进行处理以得到最终的检测结果
        # 使用计时器来测量推理时间
        with dt[1]:
            # 根据visualize变量的值来设置是否可视化。如果visualize为True，将使用increment_path函数创建一个保存可视化结果的目录
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            # 将图像传递给模型进行推理。它会返回预测结果，其中包括预测框、置信度、类别等信息。augment参数表示是否使用数据增强，visualize参数表示是否进行可视化
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        # 使用计时器来测量非最大抑制（NMS）的时间
        with dt[2]:
            # 这行代码使用非最大抑制算法对预测结果进行后处理，以去除重叠的框并保留最相关的框。它使用了一些参数，
            # 如置信度阈值（conf_thres）、IoU阈值（iou_thres）、类别信息（classes）、是否使用无关类别NMS（agnostic_nms）以及最大检测数（max_det）等
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # 负责执行第二阶段的分类器（如果使用了的话）以及处理检测预测结果
        # 暗示在此可执行第二阶段的分类器。根据注释，这是一个可选步骤，可能涉及将分类器应用于预测结果
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        # 作用是对检测结果进行处理，包括在图像上绘制框和标签，保存标签文件以及保存裁剪图像文件。根据需求，它可以在图像上绘制检测结果，保存图像和标签，以及进行一些后处理操作
        # 这个循环遍历每张图像的预测结果。在每次迭代中，i表示索引，det表示一张图像的预测结果（可能包含多个检测框）
        for i, det in enumerate(pred):  # per image
            # 增加已处理的图像数量
            seen += 1
            if webcam:  # batch_size >= 1
                # 如果是摄像头模式（webcam为True），则对批量中的每张图像进行处理，提取路径、原始图像和帧数等信息，并将这些信息记录在s变量中
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                # 如果不是摄像头模式，那么将直接提取路径、原始图像和帧数等信息
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            # 将路径p转换为Path对象
            p = Path(p)  # to Path
            # 构建保存图像文件的路径，其中save_dir是保存目录，p.name是文件名
            save_path = str(save_dir / p.name)  # im.jpg
            # 构建保存标签文件的路径，其中save_dir是保存目录，p.stem是文件名的主干部分，dataset.mode是数据集模式（可能是'image'或'video'），frame是帧数
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            # 将图像的宽高信息添加到字符串s中
            s += '%gx%g ' % im.shape[2:]  # print string
            # 构建一个张量gn，表示归一化增益，用于从图像尺寸转换到归一化尺寸
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            # 如果save_crop为True，将原始图像复制给imc，否则直接赋值
            imc = im0.copy() if save_crop else im0  # for save_crop
            # 创建一个注释器对象annotator，用于在图像上绘制边界框和标签。line_width表示线的宽度，example是类别名称的示例
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            # 如果检测结果不为空
            if len(det):
                # Rescale boxes from img_size to im0 size
                # 将检测框从im的尺寸缩放到原始图像im0的尺寸
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                # 遍历不同的类别，统计每个类别的检测数量，并将结果添加到字符串s中
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                # 遍历每个检测结果，对每个框进行处理
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            # 在图像上展示检测结果，以实时流式方式显示。如果您需要实时查看目标检测结果，这部分代码会非常有用
            # 获取经过注释器处理后的图像，即带有绘制的边界框和标签的图像
            im0 = annotator.result()
            # 如果启用了图像查看
            if view_img:
                # 件判断用于检查当前操作系统是否为Linux，并且窗口名称（p）是否已经存在于windows列表中
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    # 如果是Linux系统，并且窗口名称不在windows列表中，那么创建一个带有窗口名称的OpenCV窗口。这允许在Linux上调整窗口大小
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    # 调整OpenCV窗口的大小，以适应图像的尺寸
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                # 在OpenCV窗口中显示图像
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            # 保存带有检测结果的图像，并打印出推理时间和一些信息。如果需要将检测结果保存为图像或视频文件，以及打印出相关信息
            # 如果启用了图像保存
            if save_img:
                # 果数据集模式为'image'，表示单张图像模式
                if dataset.mode == 'image':
                    # 使用OpenCV将图像im0保存为文件，文件路径为save_path
                    cv2.imwrite(save_path, im0)
                # 否则，数据集模式可能是'video'或'stream'
                else:  # 'video' or 'stream'
                    # 如果当前视频路径vid_path[i]与保存路径save_path不同，表示开始处理新的视频
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        # 如果之前已经存在视频写入对象vid_writer[i]，则先释放它
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        # 如果是视频（vid_cap存在），获取视频的帧率、宽度和高度
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        # 如果是流（vid_cap不存在），设置默认帧率和尺寸
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        # 保存路径更改为以'.mp4'为后缀的新路径，强制使用MP4格式保存结果视频
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        # 创建一个新的视频写入对象vid_writer[i]，使用MP4格式、指定帧率和尺寸
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    # 将图像im0写入视频写入对象vid_writer[i]中，以生成视频
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        # 打印信息，包括检测的类别和数量，推理时间等
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    # 这行代码计算出每个阶段的平均速度，单位为毫秒（ms），根据每个阶段的计时器和已处理图像数量
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    # 打印出推理速度的信息，包括预处理时间、推理时间和NMS时间
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        # 如果启用了保存文本，这行代码统计已保存的文本文件数量，并生成相应的提示信息
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")

    '''
    在YOLOv5的detect函数中，update参数的作用如下：
        当update=True时：会在执行推理的过程中，根据需要对模型进行更新，以解决可能的源代码变更警告（SourceChangeWarning）问题。
            模型更新可能涉及去除不必要的优化器状态，以确保模型在推理过程中正常工作，而不受优化器状态的影响。这可以帮助避免在推理时出现不必要的错误。
        当update=False时：推理过程中不会对模型进行更新，优化器状态保持不变。这可能会在一些情况下导致模型在推理过程中出现问题，特别是当源代码发生变更时，可能会出现一些警告或错误。
    
    << SourceChangeWarning >>
    在Python中，SourceChangeWarning是一种警告类型，用于指示源代码的改变可能会导致某些操作或行为不再适用或不再具有预期的效果。
        这种警告通常是由于Python解释器的改变、库的更新或源代码的修改引起的。
    
    当Python解释器或库的开发人员在新版本中进行了一些更改，并且这些更改可能会影响到旧版本代码的正确性或行为时，就会触发SourceChangeWarning警告。  
        这种警告的目的是提醒开发人员在更新库或源代码时要注意潜在的问题，以便及时修复或调整代码以适应变更
        
    对于YOLOv5中的detect函数，update参数的作用是为了避免在推理过程中由于源代码变更而引发的警告或错误。
        通过在推理过程中更新模型，可以确保模型的状态与源代码变更保持一致，从而减少可能的问题。这样做可以帮助开发人员在进行目标检测推理时获得更稳定的结果。
    '''
    # 如果启用了更新操作
    if update:
        # 调用strip_optimizer函数，用于更新模型以解决潜在的源代码变更警告（SourceChangeWarning）问题
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    # parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    # parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) datasets.yaml path')
    # parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    # parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    # parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    # parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    # parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--view-img', action='store_true', help='show results')
    # parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    # parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    # parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    # parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    # parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    # parser.add_argument('--augment', action='store_true', help='augmented inference')
    # parser.add_argument('--visualize', action='store_true', help='visualize features')
    # parser.add_argument('--update', action='store_true', help='update all models')
    # parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    # parser.add_argument('--name', default='exp', help='save results to project/name')
    # parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    # parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    # parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    # parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    # parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    # parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    # parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/train/5mNAMC3/weights/best_openvino_model_int8', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=r'D:\StuData\tomato\dataset_factory\background\coco_2\images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) datasets.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
