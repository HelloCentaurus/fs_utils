#!/usr/bin/env python
"""
Copyright (c) 2019, by the Authors: Amir H. Abdi
This script is freely available under the MIT Public License.
Please see the License file in the root for details.

The following code snippet will convert the keras model files
to the freezed .pb tensorflow weight file. The resultant TensorFlow model
holds both the model architecture and its associated weights.
"""

import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from pathlib import Path
from absl import logging
import keras
from keras import backend as K
from keras.models import model_from_json, model_from_yaml

K.set_learning_phase(0)

class H5_2_Pb:
    def __init__(self, input_model, output_model):
        self.input_model = input_model        # Path to the input model
        self.input_model_json = None   # Path to the input model architecture in json format
        self.input_model_yaml = None   # Path to the input model architecture in yaml format
        self.output_model = output_model       # Path where the converted model will be stored
        self.save_graph_def = False    # Whether to save the graphdef.pbtxt file which contains the graph definition in ASCII format
        self.output_nodes_prefix = None
        self.quantize = False
        self.channels_first = False
        self.output_meta_ckpt = False

    def load_model(self, input_model_path, input_json_path=None, input_yaml_path=None):
        if not Path(input_model_path).exists():
            raise FileNotFoundError(
                'Model file `{}` does not exist.'.format(input_model_path))
        try:
            model = keras.models.load_model(input_model_path)
            return model
        except FileNotFoundError as err:
            logging.error('Input mode file (%s) does not exist.', self.input_model)
            raise err
        except ValueError as wrong_file_err:
            if input_json_path:
                if not Path(input_json_path).exists():
                    raise FileNotFoundError(
                        'Model description json file `{}` does not exist.'.format(
                            input_json_path))
                try:
                    model = model_from_json(open(str(input_json_path)).read())
                    model.load_weights(input_model_path)
                    return model
                except Exception as err:
                    logging.error("Couldn't load model from json.")
                    raise err
            elif input_yaml_path:
                if not Path(input_yaml_path).exists():
                    raise FileNotFoundError(
                        'Model description yaml file `{}` does not exist.'.format(
                            input_yaml_path))
                try:
                    model = model_from_yaml(open(str(input_yaml_path)).read())
                    model.load_weights(input_model_path)
                    return model
                except Exception as err:
                    logging.error("Couldn't load model from yaml.")
                    raise err
            else:
                logging.error(
                    'Input file specified only holds the weights, and not '
                    'the model definition. Save the model using '
                    'model.save(filename.h5) which will contain the network '
                    'architecture as well as its weights. '
                    'If the model is saved using the '
                    'model.save_weights(filename) function, either '
                    'input_model_json or input_model_yaml flags should be set to '
                    'to import the network architecture prior to loading the '
                    'weights. \n'
                    'Check the keras documentation for more details '
                    '(https://keras.io/getting-started/faq/)')
                raise wrong_file_err


    def main(self):
        # If output_model path is relative and in cwd, make it absolute from root
        output_model = self.output_model
        if str(Path(output_model).parent) == '.':
            output_model = str((Path.cwd() / output_model))

        output_fld = Path(output_model).parent
        output_model_name = Path(output_model).name
        output_model_stem = Path(output_model).stem
        output_model_pbtxt_name = output_model_stem + '.pbtxt'

        # Create output directory if it does not exist
        Path(output_model).parent.mkdir(parents=True, exist_ok=True)

        if self.channels_first:
            K.set_image_data_format('channels_first')
        else:
            K.set_image_data_format('channels_last')

        model = self.load_model(self.input_model, self.input_model_json, self.input_model_yaml)

        # TODO(amirabdi): Support networks with multiple inputs
        orig_output_node_names = [node.op.name for node in model.outputs]
        if self.output_nodes_prefix:
            num_output = len(orig_output_node_names)
            pred = [None] * num_output
            converted_output_node_names = [None] * num_output

            # Create dummy tf nodes to rename output
            for i in range(num_output):
                converted_output_node_names[i] = '{}{}'.format(self.output_nodes_prefix, i)
                pred[i] = tf.identity(model.outputs[i], name=converted_output_node_names[i])
        else:
            converted_output_node_names = orig_output_node_names
        logging.info('Converted output node names are: %s', str(converted_output_node_names))

        sess = K.get_session()
        if self.output_meta_ckpt:
            saver = tf.train.Saver()
            saver.save(sess, str(output_fld / output_model_stem))

        if self.save_graph_def:
            tf.train.write_graph(sess.graph.as_graph_def(), str(output_fld), output_model_pbtxt_name, as_text=True)
            logging.info('Saved the graph definition in ascii format at %s', str(Path(output_fld) / output_model_pbtxt_name))

        if self.quantize:
            from tensorflow.tools.graph_transforms import TransformGraph
            transforms = ["quantize_weights", "quantize_nodes"]
            transformed_graph_def = TransformGraph(sess.graph.as_graph_def(), [], converted_output_node_names, transforms)
            constant_graph = graph_util.convert_variables_to_constants(
                sess,
                transformed_graph_def,
                converted_output_node_names)
        else:
            constant_graph = graph_util.convert_variables_to_constants(
                sess,
                sess.graph.as_graph_def(),
                converted_output_node_names)

        graph_io.write_graph(constant_graph, str(output_fld), output_model_name, as_text=False)
        logging.info('Saved the freezed graph at %s', str(Path(output_fld) / output_model_name))

if __name__ == "__main__":
    # sys.arg['input_model'] ='./models/fsnet_001.h5'
    # sys.arg['output_model'] ='./models/fsnet_001.pb'
    convert = H5_2_Pb(input_model='../Classify_Nets/model_datas/IpdaModel_1206-4.h5', output_model='./models/IpdaModel_1206-4.pb')
    convert.main()