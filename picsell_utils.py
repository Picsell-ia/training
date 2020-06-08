import json
import tensorflow as tf
from object_detection.utils import dataset_util
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from PIL import Image, ImageDraw
import os
import numpy as np 
import cv2
import functools
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from object_detection.builders import dataset_builder
from object_detection.builders import graph_rewriter_builder
from object_detection.builders import model_builder
from object_detection.legacy import trainer
from object_detection.legacy import evaluator
from object_detection import exporter
from object_detection.utils import ops as utils_ops
from object_detection.utils import visualization_utils as vis_util
from IPython.display import display
import random




        



def create_record_files(label_path, record_dir, tfExample_generator, annotation_type):
    '''
        Function used to create the TFRecord files used for the training and evaluation.
        
        TODO: Shard files for large dataset


        Args:
            label_path: Path to the label map file.
            record_dir: Path used to write the records files.
            tfExample_generator: Use the generator from the Picsell.ia SDK by default or provide your own generator.
            annotation_type: "polygon" or "rectangle", depending on your project type. Polygon will compute the masks from your polygon-type annotations. 
    '''
    label_map = label_map_util.load_labelmap(label_path)
    label_map = label_map_util.get_label_map_dict(label_map) 
    datasets = ["train", "eval"]
    
    for dataset in datasets:
        output_path = os.path.join(record_dir, dataset+".record")
        writer = tf.python_io.TFRecordWriter(output_path)
        for variables in tfExample_generator(label_map, ensemble=dataset, annotation_type=annotation_type):
            if annotation_type=="polygon":
                (width, height, xmins, xmaxs, ymins, ymaxs, filename,
                     encoded_jpg, image_format, classes_text, classes, masks) = variables
            
                tf_example = tf.train.Example(features=tf.train.Features(feature={
                    'image/height': dataset_util.int64_feature(height),
                    'image/width': dataset_util.int64_feature(width),
                    'image/filename': dataset_util.bytes_feature(filename),
                    'image/source_id': dataset_util.bytes_feature(filename),
                    'image/encoded': dataset_util.bytes_feature(encoded_jpg),
                    'image/format': dataset_util.bytes_feature(image_format),
                    'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
                    'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
                    'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
                    'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
                    'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
                    'image/object/class/label': dataset_util.int64_list_feature(classes),
                    'image/object/mask': dataset_util.bytes_list_feature(masks)
                }))               
                
            elif annotation_type=="rectangle":
                (width, height, xmins, xmaxs, ymins, ymaxs, filename,
                        encoded_jpg, image_format, classes_text, classes) = variables

                tf_example = tf.train.Example(features=tf.train.Features(feature={
                    'image/height': dataset_util.int64_feature(height),
                    'image/width': dataset_util.int64_feature(width),
                    'image/filename': dataset_util.bytes_feature(filename),
                    'image/source_id': dataset_util.bytes_feature(filename),
                    'image/encoded': dataset_util.bytes_feature(encoded_jpg),
                    'image/format': dataset_util.bytes_feature(image_format),
                    'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
                    'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
                    'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
                    'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
                    'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
                    'image/object/class/label': dataset_util.int64_list_feature(classes)
                    }))

            writer.write(tf_example.SerializeToString())    
        writer.close()
        print('Successfully created the TFRecords: {}'.format(output_path))

def update_num_classes(config_dict, label_map):
    ''' 
    Update the number of classes inside the protobuf configuration dictionnary depending on the number of classes inside the label map.

        Args :
            config_dict:  A configuration dictionnary loaded from the protobuf file with config_util.get_configs_from_pipeline_file().
            label_map: Protobuf label_map loaded with label_map_util.load_labelmap()
        Raises:
            ValueError if the backbone architecture isn't known.
    '''
    model_config = config_dict["model"]
    n_classes = len(label_map.item)
    meta_architecture = model_config.WhichOneof("model")
    if meta_architecture == "faster_rcnn":
        model_config.faster_rcnn.num_classes = n_classes
    elif meta_architecture == "ssd":
        model_config.ssd.num_classes = n_classes
    else:
        raise ValueError("Expected the model to be one of 'faster_rcnn' or 'ssd'.")


def set_image_resizer(config_dict, shape):
    '''
        Update the image resizer shapes.

        Args:
            config_dict:  A configuration dictionnary loaded from the protobuf file with config_util.get_configs_from_pipeline_file().
            shape: The new shape for the image resizer. 
                    [max_dimension, min_dimension] for the keep_aspect_ratio_resizer (default resizer for faster_rcnn backbone). 
                    [width, height] for the fixed_shape_resizer (default resizer for SSD backbone)

        Raises: 
            ValueError if the backbone architecture isn't known.
    '''

    model_config = config_dict["model"]
    meta_architecture = model_config.WhichOneof("model")
    if meta_architecture == "faster_rcnn":
        image_resizer = model_config.faster_rcnn.image_resizer
    elif meta_architecture == "ssd":
        image_resizer = model_config.ssd.image_resizer
    else:
        raise ValueError("Unknown model type: {}".format(meta_architecture))
    
    if image_resizer.HasField("keep_aspect_ratio_resizer"):
        image_resizer.keep_aspect_ratio_resizer.max_dimension = shape[1]
        image_resizer.keep_aspect_ratio_resizer.min_dimension = shape[0]

    elif image_resizer.HasField("fixed_shape_resizer"):
        image_resizer.fixed_shape_resizer.height = shape[1]
        image_resizer.fixed_shape_resizer.width = shape[0]

def edit_eval_config(config_dict, annotation_type, eval_number):
    '''
        Update the eval_config protobuf message from a config_dict.
        Checks if the metrics_set is the right one then update the evaluation number. 

        Args:
            config_dict: A configuration dictionnary loaded from the protobuf file with config_util.get_configs_from_pipeline_file().
            annotation_type: Should be either "rectangle" or "polygon". Depends on your project type. 
            eval_number: The number of images you want to run your evaluation on. 
        
        Raises:
            ValueError Wrong annotation type provided. If you didn't provide the right annotation_type
            ValueError "eval_number isn't an int". If you didn't provide a int for the eval_number.
    '''


    eval_config = config_dict["eval_config"]
    eval_config.num_visualizations = 0
    if annotation_type=="rectangle":
        eval_config.metrics_set[0] = "coco_detection_metrics"
    elif annotation_type=="polygon":
        eval_config.metrics_set[0] = "coco_mask_metrics"
    else:
        raise ValueError("Wrong annotation type provided")
    if isinstance(eval_number, int):
        eval_config.num_examples = eval_number
    else: 
        raise ValueError("eval_number isn't an int")


def update_different_paths(config_dict, ckpt_path, label_map_path, train_record_path, eval_record_path):
    '''
        Update the different paths required for the whole configuration.

    Args: 
        config_dict: A configuration dictionnary loaded from the protobuf file with config_util.get_configs_from_pipeline_file().
        ckpt_path: Path to your checkpoint. 
        label_map_path: Path to your label map.
        train_record_path: Path to your train record file.
        eval_record_path: Path to your eval record file.

    '''
    config_dict["train_config"].fine_tune_checkpoint = ckpt_path
    config_util._update_label_map_path(config_dict, label_map_path)
    config_util._update_tf_record_input_path(config_dict["train_input_config"], train_record_path)
    config_util._update_tf_record_input_path(config_dict["eval_input_config"], eval_record_path)


def edit_masks(config_dict, mask_type="PNG_MASKS"):
    """
        Update the configuration to take into consideration the right mask_type. By default we record mask as "PNG_MASKS".

        Args:
            config_dict: A configuration dictionnary loaded from the protobuf file with config_util.get_configs_from_pipeline_file().
            mask_type: String name to identify mask type, either "PNG_MASKS" or "NUMERICAL_MASKS"
        Raises:
            ValueError if the mask type isn't known.
    """

    config_dict["train_input_config"].load_instance_masks = True
    config_dict["eval_input_config"].load_instance_masks = True
    if mask_type=="PNG_MASKS":
        config_dict["train_input_config"].mask_type = 2
        config_dict["eval_input_config"].mask_type = 2
    elif mask_type=="NUMERICAL_MASKS":
        config_dict["train_input_config"].mask_type = 1
        config_dict["eval_input_config"].mask_type = 1
    else:
        raise ValueError("Wrong Mask type provided")

def edit_config(model_selected, config_output_dir, num_steps, label_map_path, record_dir, eval_number, annotation_type, 
                batch_size=None, learning_rate=None, resizer_size=None):
    '''
        Wrapper to edit the essential values inside the base configuration protobuf file provided with an object-detection/segmentation checkpoint.
        This configuration file is what will entirely define your model, pre-processing, training, evaluation etc. It is the most important file of a model with the checkpoint file and should never be deleted. 
        This is why it is saved in almost every directory where you did something to keep redondancy but also to be sure to have the right config file used at this moment.
        For advanced users, if you want to dwell deep inside the configuration file you should read the proto definitions inside the proto directory of the object-detection API.

        Args: 
            Required:
                model_selected: The checkpoint you want to resume from.
                config_output_dir: The path where you want to save your edited protobuf configuration file.
                num_steps: The number of steps you want to train on.
                label_map_path: The path to your label_map.pbtxt file.
                record_dir: The path to the directory where your TFRecord files are saved.
                eval_number: The number of images you want to evaluate on.
                annotation_type: Should be either "rectangle" or "polygon", depending on how you annotated your images.

            Optional:
                batch_size: The batch size you want to use. If not provided it will use the previous one. 
                learning_rate: The learning rate you want to use for the training. If not provided it will use the previous one. 
                                Please see config_utils.update_initial_learning_rate() inside the object_detection folder for indepth details on what happens when updating it.
                resizer_size: The shape used to update your image resizer. Please see set_image_resizer() for more details on this. If not provided it will use the previous one.            

    '''

    
    file_list = os.listdir(model_selected)
    ckpt_ids = []
    for p in file_list:
        if "index" in p:
            if "-" in p:
                ckpt_ids.append(int(p.split('-')[1].split('.')[0]))
    if len(ckpt_ids)>0:
        ckpt_path = os.path.join(model_selected,"model.ckpt-{}".format(str(max(ckpt_ids))))
    
    else:
        ckpt_path = os.path.join(model_selected, "model.ckpt")

    configs = config_util.get_configs_from_pipeline_file(os.path.join(model_selected,'pipeline.config'))
    label_map = label_map_util.load_labelmap(label_map_path)

    config_util._update_train_steps(configs, num_steps)
    update_different_paths(configs, ckpt_path=ckpt_path, 
                            label_map_path=label_map_path, 
                            train_record_path=os.path.join(record_dir, "train.record"), 
                            eval_record_path=os.path.join(record_dir,"eval.record"))

    if learning_rate is not None:
        config_util._update_initial_learning_rate(configs, learning_rate)

    if batch_size is not None:
        config_util._update_batch_size(configs, batch_size)

    if annotation_type=="polygon":
        edit_masks(configs, mask_type="PNG_MASKS")
   
    if resizer_size is not None:
        set_image_resizer(configs, resizer_size)

    edit_eval_config(configs, annotation_type, eval_number)
    update_num_classes(configs, label_map)
    config_proto = config_util.create_pipeline_proto_from_configs(configs)
    config_util.save_pipeline_config(config_proto, directory=config_output_dir)


def train(master='', save_summaries_secs=30, task=0, num_clones=1, clone_on_cpu=False, worker_replicas=1, ps_tasks=0, 
                    ckpt_dir='', conf_dir='', train_config_path='', input_config_path='', model_config_path=''):   
   

    pipeline_config_path = os.path.join(conf_dir,"pipeline.config")
    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)

    tf.logging.set_verbosity(tf.logging.INFO)
    assert ckpt_dir, '`ckpt_dir` is missing.'
    if task == 0: tf.gfile.MakeDirs(ckpt_dir)
    if pipeline_config_path:
        configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
        if task == 0:
            tf.gfile.Copy(pipeline_config_path,
                          os.path.join(ckpt_dir, 'pipeline.config'),
                          overwrite=True)
    else:
        configs = config_util.get_configs_from_multiple_files(
                        model_config_path=model_config_path,
                        train_config_path=train_config_path,
                        train_input_config_path=input_config_path)
        if task == 0:
            for name, config in [('model.config', model_config_path),
                                ('train.config', train_config_path),
                                ('input.config', input_config_path)]:
                tf.gfile.Copy(config, os.path.join(ckpt_dir, name),
                            overwrite=True)

    model_config = configs['model']
    train_config = configs['train_config']
    input_config = configs['train_input_config']

    model_fn = functools.partial(
        model_builder.build,
        model_config=model_config,
        is_training=True)

    def get_next(config):
        return dataset_builder.make_initializable_iterator(
            dataset_builder.build(config)).get_next()

    create_input_dict_fn = functools.partial(get_next, input_config)

    env = json.loads(os.environ.get('TF_CONFIG', '{}'))
    cluster_data = env.get('cluster', None)
    cluster = tf.train.ClusterSpec(cluster_data) if cluster_data else None
    task_data = env.get('task', None) or {'type': 'master', 'index': 0}
    task_info = type('TaskSpec', (object,), task_data)

    # Parameters for a single worker.
    ps_tasks = 0
    worker_replicas = 1
    worker_job_name = 'lonely_worker'
    task = 0
    is_chief = True
    master = ''

    if cluster_data and 'worker' in cluster_data:
    # Number of total worker replicas include "worker"s and the "master".
        worker_replicas = len(cluster_data['worker']) + 1
    if cluster_data and 'ps' in cluster_data:
        ps_tasks = len(cluster_data['ps'])

    if worker_replicas > 1 and ps_tasks < 1:
        raise ValueError('At least 1 ps task is needed for distributed training.')

    if worker_replicas >= 1 and ps_tasks > 0:
    # Set up distributed training.
        server = tf.train.Server(tf.train.ClusterSpec(cluster), protocol='grpc',
                                job_name=task_info.type,
                                task_index=task_info.index)
        if task_info.type == 'ps':
            server.join()
            return

        worker_job_name = '%s/task:%d' % (task_info.type, task_info.index)
        task = task_info.index
        is_chief = (task_info.type == 'master')
        master = server.target

    graph_rewriter_fn = None
    if 'graph_rewriter_config' in configs:
        graph_rewriter_fn = graph_rewriter_builder.build(
            configs['graph_rewriter_config'], is_training=True)

    trainer.train(
        create_input_dict_fn,
        model_fn,
        train_config,
        master,
        task,
        num_clones,
        worker_replicas,
        clone_on_cpu,
        ps_tasks,
        worker_job_name,
        is_chief,
        ckpt_dir,
        graph_hook_fn=graph_rewriter_fn,
        save_summaries_secs=save_summaries_secs)

def evaluate(eval_dir, config_dir, checkpoint_dir, eval_training_data=False):

    '''
        Function used to evaluate your trained model. 

        Args: 
            Required:               
                eval_dir: The directory where the tfevent file will be saved.
                config_dir: The protobuf configuration directory.
                checkpoint_dir: The directory where the checkpoint you want to evaluate is.
            
            Optional:
                eval_training_data: Is set to True the evaluation will be run on the training dataset.

        Returns:
            A dictionnary of metrics ready to be sent to the picsell.ia platform.
    '''

    tf.reset_default_graph()
    tf.gfile.MakeDirs(eval_dir)
    configs = config_util.get_configs_from_pipeline_file(os.path.join(config_dir, "pipeline.config"))
    model_config = configs['model']
    eval_config = configs['eval_config']
    input_config = configs['eval_input_config']
    if eval_training_data:
        input_config = configs['train_input_config']

    model_fn = functools.partial(model_builder.build, model_config=model_config, is_training=False)

    def get_next(config):
        return dataset_builder.make_initializable_iterator(dataset_builder.build(config)).get_next()

    create_input_dict_fn = functools.partial(get_next, input_config)

    categories = label_map_util.create_categories_from_labelmap(
      input_config.label_map_path)

    eval_config.max_evals = 1

    graph_rewriter_fn = None
    if 'graph_rewriter_config' in configs:
        graph_rewriter_fn = graph_rewriter_builder.build(
            configs['graph_rewriter_config'], is_training=False)

    metrics = evaluator.evaluate(
          create_input_dict_fn,
          model_fn,
          eval_config,
          categories,
          checkpoint_dir,
          eval_dir,
          graph_hook_fn=graph_rewriter_fn)
    return {k:str(round(v, 3)) for k,v in metrics.items()}



def tfevents_to_dict(path):
    '''Get a dictionnary of scalars from the tfevent inside the training directory.

        Args: 
            path: The path to the training directory where a tfevent file is saved.
        
        Returns:
            A dictionnary of scalars logs.
    '''
    event = [filename for filename in os.listdir(path) if filename.startswith("events.out")][-1]
    event_acc = EventAccumulator(os.path.join(path,event)).Reload()
    logs = dict()
    # dict_logs = dict()
    # K=0
    for scalar_key in event_acc.scalars.Keys():
        scalar_dict = {"step": [], "value": []}

        for scalars in event_acc.Scalars(scalar_key):
            scalar_dict["step"].append(scalars.step)
            scalar_dict["value"].append(scalars.value)
        logs[scalar_key] = scalar_dict
    #     if K==0:
    #         K = len(event_acc.Scalars(scalar_key))
    #         if K<500:
    #             Idx = np.logspace(1,np.log10(K-1), int(K/10), dtype=np.uint32)
    #             Idx = list(np.arange(0,9,1))+list(Idx)

    # if len(Idx)<=len(logs["Losses/TotalLoss"]["value"]):
    #     for key in logs.keys():
    #         dict_logs[key]=dict.fromkeys(["value","step"])
    #         dict_logs[key]["value"] = list(np.array(logs[key]["value"])[Idx])
    #         dict_logs[key]["step"] = list(np.array(logs[key]["step"])[Idx].astype(float))
    #     return dict_logs

    return logs
    


def export_infer_graph(ckpt_dir, exported_model_dir, pipeline_config_path,
                        write_inference_graph=False, input_type="image_tensor", input_shape=None):
    
    ''' Export your checkpoint to a saved_model.pb file
        Args:
            Required:
                ckpt_dir: The directory where your checkpoint to export is located.
                exported_model_dir: The directory where you want to save your model.
                pipeline_çonfig_path: The directory where you protobuf configuration is located.

    '''

    tf.reset_default_graph()
    pipeline_config_path = os.path.join(pipeline_config_path,"pipeline.config")
    config_dict = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    ckpt_number = str(config_dict["train_config"].num_steps)
    pipeline_config = config_util.create_pipeline_proto_from_configs(config_dict)

    trained_checkpoint_prefix = os.path.join(ckpt_dir,'model.ckpt-'+ckpt_number)
    exporter.export_inference_graph(
        input_type, pipeline_config, trained_checkpoint_prefix,
        exported_model_dir, input_shape=input_shape,
        write_inference_graph=write_inference_graph)



def infer(path_list, exported_model_dir, label_map_path, results_dir, disp=True, num_infer=5, min_score_thresh=0.7):

    ''' Use your exported model to infer on a path list of images. 

        Args:
            Required:
                path_list: A list of images paths to infer on.
                exported_model_dir: The path used to saved your model.
                label_mapt_path: The path to your label_map file.
                results_dir: The directory where you want to save your infered images.

            Optional:
                disp: Set to false if you are not in an interactive python environment. Will display image in the environment if set to True.
                num_infer: The number of images you want to infer on. 
                min_score_tresh: The minimal confidence treshold to keep the detection.

    '''
    saved_model_path = os.path.join(exported_model_dir, "saved_model")
    predict_fn = tf.contrib.predictor.from_saved_model(saved_model_path)
    random.shuffle(path_list)
    path_list = path_list[:num_infer]
    with tf.Session() as sess:
        category_index = label_map_util.create_category_index_from_labelmap(label_map_path)
        for img_path in path_list:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = np.expand_dims(img, 0)
            output_dict = predict_fn({"inputs": img_tensor})
            
           

            num_detections = int(output_dict.pop('num_detections'))
            output_dict = {key:value[0, :num_detections] for key,value in output_dict.items()}
            output_dict['num_detections'] = num_detections
            output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

            if 'detection_masks' in output_dict:
                # Reframe the the bbox mask to the image size.
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                                                            output_dict['detection_masks'], 
                                                            output_dict['detection_boxes'],
                                                            img.shape[0], 
                                                            img.shape[1])      
                detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)          
                mask_refr = sess.run(detection_masks_reframed)
                output_dict['detection_masks_reframed'] = mask_refr
            masks = output_dict.get('detection_masks_reframed', None)
            boxes = output_dict["detection_boxes"]
            classes = output_dict["detection_classes"]
            scores = output_dict["detection_scores"]
            

            b = []
            c = []
            s = []
            m = []
            k = 0
            for classe in classes:
                b.append(boxes[k])
                c.append(classe)
                s.append(scores[k])
                if masks is not None:
                    m.append(masks[k])
                k+=1
            boxes = np.array(b)
            classes = np.array(c)
            scores = np.array(s)
            if masks is not None:
                masks = np.array(m)


            vis_util.visualize_boxes_and_labels_on_image_array(img, 
                                                boxes,
                                                classes,
                                                scores,
                                                category_index,
                                                instance_masks=masks,
                                                use_normalized_coordinates=True,
                                                line_thickness=3,
                                                min_score_thresh=min_score_thresh,
                                                max_boxes_to_draw=None)

            img_name = img_path.split("/")[-1]
            Image.fromarray(img).save(os.path.join(results_dir,img_name))
            
            if disp == True:
                display(Image.fromarray(img))




