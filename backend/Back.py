# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from flask_socketio import SocketIO, emit
# import os
# import logging
# import json
# import base64
# from train import run

# app = Flask(__name__)
# CORS(app)
# socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')



# # Setup logging
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

# @app.route('/save-annotations', methods=['POST'])
# def save_annotations():
#     try:
#         data = request.json
#         annotations = data['annotations']
#         dataset_folder = data['datasetFolder']
#         train_percent = data['trainPercent']
#         val_percent = data['valPercent']
#         test_percent = data['testPercent']
#         tags = data['tags']

#         base_dir = os.path.join('C:\\Users\\admin\\Desktop\\Nommas\\image_annotator-master\\python\\yolov5', dataset_folder)

#         try:
#             os.makedirs(os.path.join(base_dir, 'train', 'images'), exist_ok=True)
#             os.makedirs(os.path.join(base_dir, 'train', 'labels'), exist_ok=True)
#             os.makedirs(os.path.join(base_dir, 'valid', 'images'), exist_ok=True)
#             os.makedirs(os.path.join(base_dir, 'valid', 'labels'), exist_ok=True)
#             os.makedirs(os.path.join(base_dir, 'test', 'images'), exist_ok=True)
#             os.makedirs(os.path.join(base_dir, 'test', 'labels'), exist_ok=True)

#             train, val, test = [], [], []

#             for item in annotations:
#                 label = item['label']
#                 if label not in tags:
#                     continue

#                 rand = os.urandom(1)[0] / 255.0
#                 if rand < float(train_percent) / 100:
#                     train.append(item)
#                 elif rand < (float(train_percent) + float(val_percent)) / 100:
#                     val.append(item)
#                 else:
#                     test.append(item)

#             def save_annotations(data, type):
#                 for anno in data:
#                     image_id = anno['imageId']
#                     label = anno['label']
#                     try:
#                         label_index = tags.index(label)
#                     except ValueError:
#                         continue
#                     x_center = anno['x_center'] / 100
#                     y_center = anno['y_center'] / 100
#                     bbox_width = anno['bbox_width'] / 100
#                     bbox_height = anno['bbox_height'] / 100
#                     image_data = anno['imageData']

#                     image_filename = os.path.basename(image_id) + '.png'
#                     label_filename = os.path.basename(image_id) + '.txt'

#                     label_file = os.path.join(base_dir, type, 'labels', label_filename)
#                     with open(label_file, 'a') as f:
#                         annotation_str = f"{label_index} {x_center} {y_center} {bbox_width} {bbox_height}\n"
#                         f.write(annotation_str)

#                     image_file = os.path.join(base_dir, type, 'images', image_filename)
#                     with open(image_file, 'wb') as f:
#                         try:
#                             if image_data.startswith('data:image'):
#                                 f.write(base64.b64decode(image_data.split(',')[1]))
#                         except Exception as e:
#                             continue

#             save_annotations(train, 'train')
#             save_annotations(val, 'valid')
#             save_annotations(test, 'test')

#             data_yaml = f"""
#             train: {os.path.join(base_dir, 'train', 'images').replace('\\', '/')}
#             val: {os.path.join(base_dir, 'valid', 'images').replace('\\', '/')}
#             nc: {len(tags)}
#             names: {json.dumps(tags)}
#             """

#             data_yaml_path = os.path.join(base_dir, 'data.yaml')
#             with open(data_yaml_path, 'w') as f:
#                 f.write(data_yaml)

#             return jsonify(message='Annotations saved and training data prepared successfully.', data_yaml_path=data_yaml_path)
#         except Exception as e:
#             return jsonify(error=f'Failed to save annotations. {str(e)}'), 500
#     except Exception as e:
#         return jsonify(error=f'Failed to parse request data. {str(e)}'), 500




# @app.route('/start-training', methods=['GET'])
# def start_training():
#     dataset_folder = request.args.get('dataset_folder')
#     data_yaml_path = os.path.join('C:\\Users\\admin\\Desktop\\Nommas\\image_annotator-master\\python\\yolov5', dataset_folder, 'data.yaml')
#     logger.info(f"Starting training with data: {data_yaml_path}")

#     # Run the training script and emit progress via Socket.IO
#     def run_training(data_path):
#         try:
#             run(data=data_path, weights='yolov5s.pt', epochs=1, batch_size=2, imgsz=640, callbacks=socketio.emit)
#             socketio.emit('training_completed')
#             return True, "Training completed successfully."
#         except Exception as e:
#             logger.error(str(e))
#             return False, str(e)
    
#     success, output = run_training(data_yaml_path)
    
#     if success:
#         return jsonify(message='Training started successfully.')
#     else:
#         return jsonify(error='Training failed.', details=output), 500

# if __name__ == "__main__":
#     socketio.run(app, host="0.0.0.0", port=5000)from flask import Flask, request, jsonifyfrom flask import Flask, request, jsonify
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
import json
import base64
from train import run

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@app.route('/save-annotations', methods=['POST'])
def save_annotations():
    try:
        data = request.json
        annotations = data['annotations']
        dataset_folder = data['datasetFolder']
        train_percent = data['trainPercent']
        val_percent = data['valPercent']
        test_percent = data['testPercent']
        tags = data['tags']

        base_dir = os.path.join(dataset_folder)
        os.makedirs(os.path.join(base_dir, 'train', 'images'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'train', 'labels'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'valid', 'images'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'valid', 'labels'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'test', 'images'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'test', 'labels'), exist_ok=True)

        train, val, test = [], [], []

        for item in annotations:
            label = item['label']
            if label not in tags:
                continue

            rand = os.urandom(1)[0] / 255.0
            if rand < float(train_percent) / 100:
                train.append(item)
            elif rand < (float(train_percent) + float(val_percent)) / 100:
                val.append(item)
            else:
                test.append(item)

        def save_annotations(data, type):
            for anno in data:
                image_id = anno['imageId']
                label = anno['label']
                try:
                    label_index = tags.index(label)
                except ValueError:
                    continue
                x_center = anno['x_center'] / 100
                y_center = anno['y_center'] / 100
                bbox_width = anno['bbox_width'] / 100
                bbox_height = anno['bbox_height'] / 100
                image_data = anno['imageData']

                image_filename = os.path.basename(image_id) + '.png'
                label_filename = os.path.basename(image_id) + '.txt'

                label_file = os.path.join(base_dir, type, 'labels', label_filename)
                with open(label_file, 'a') as f:
                    annotation_str = f"{label_index} {x_center} {y_center} {bbox_width} {bbox_height}\n"
                    f.write(annotation_str)

                image_file = os.path.join(base_dir, type, 'images', image_filename)
                with open(image_file, 'wb') as f:
                    try:
                        if image_data.startswith('data:image'):
                            f.write(base64.b64decode(image_data.split(',')[1]))
                    except Exception as e:
                        continue

        save_annotations(train, 'train')
        save_annotations(val, 'valid')
        save_annotations(test, 'test')

        data_yaml = f"""
        train: {os.path.join(base_dir, 'train', 'images').replace('\\', '/')}
        val: {os.path.join(base_dir, 'valid', 'images').replace('\\', '/')}
        nc: {len(tags)}
        names: {json.dumps(tags)}
        """

        data_yaml_path = os.path.join(base_dir, 'data.yaml')
        with open(data_yaml_path, 'w') as f:
            f.write(data_yaml)

        return jsonify(message='Annotations saved and training data prepared successfully.', data_yaml_path=data_yaml_path)
    except Exception as e:
        return jsonify(error=f'Failed to parse request data. {str(e)}'), 500

@app.route('/start-training', methods=['GET'])
def start_training():
    dataset_folder = request.args.get('dataset_folder')
    data_yaml_path = os.path.join( dataset_folder, 'data.yaml')
    logger.info(f"Starting training with data: {data_yaml_path}")

    # Run the training script without emitting progress
    def run_training(data_path):
        try:
            run(data=data_path, weights='yolov5s.pt', epochs=1, batch_size=2, imgsz=640, callbacks=None)
            return True, "Training completed successfully."
        except Exception as e:
            logger.error(str(e))
            return False, str(e)

    success, output = run_training(data_yaml_path)
    
    if success:
        return jsonify(message='Training started successfully.')
    else:
        return jsonify(error='Training failed.', details=output), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
