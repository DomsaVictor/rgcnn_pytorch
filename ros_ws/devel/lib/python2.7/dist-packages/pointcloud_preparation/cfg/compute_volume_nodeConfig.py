## *********************************************************
##
## File autogenerated for the pointcloud_preparation package
## by the dynamic_reconfigure package.
## Please do not edit.
##
## ********************************************************/

from dynamic_reconfigure.encoding import extract_params

inf = float('inf')

config_description = {'upper': 'DEFAULT', 'lower': 'groups', 'srcline': 246, 'name': 'Default', 'parent': 0, 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'cstate': 'true', 'parentname': 'Default', 'class': 'DEFAULT', 'field': 'default', 'state': True, 'parentclass': '', 'groups': [], 'parameters': [{'srcline': 291, 'description': 'Select camera', 'max': 2.0, 'cconsttype': 'const double', 'ctype': 'double', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'name': 'selection_camera', 'edit_method': '', 'default': 1.0, 'level': 1, 'min': 1.0, 'type': 'double'}, {'srcline': 291, 'description': 'Nr_points_input_pointcloud', 'max': 20000.0, 'cconsttype': 'const double', 'ctype': 'double', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'name': 'nr_points_initial', 'edit_method': '', 'default': 100.0, 'level': 1, 'min': 0.0, 'type': 'double'}, {'srcline': 291, 'description': 'Nr_points_input_pointcloud', 'max': 100.0, 'cconsttype': 'const double', 'ctype': 'double', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'name': 'dividing_number', 'edit_method': '', 'default': 3.0, 'level': 1, 'min': 1.0, 'type': 'double'}, {'srcline': 291, 'description': 'Perpendicular threshold', 'max': 1.0, 'cconsttype': 'const double', 'ctype': 'double', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'name': 'perpendicular_threshold', 'edit_method': '', 'default': 0.01, 'level': 1, 'min': 0.001, 'type': 'double'}, {'srcline': 291, 'description': 'Parallel threshold', 'max': 1.0, 'cconsttype': 'const double', 'ctype': 'double', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'name': 'parallel_threshold', 'edit_method': '', 'default': 0.01, 'level': 1, 'min': 0.001, 'type': 'double'}, {'srcline': 291, 'description': 'Threshold X', 'max': 0.1, 'cconsttype': 'const double', 'ctype': 'double', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'name': 'threshold_x', 'edit_method': '', 'default': 0.002, 'level': 1, 'min': 0.001, 'type': 'double'}, {'srcline': 291, 'description': 'Threshold Y', 'max': 0.1, 'cconsttype': 'const double', 'ctype': 'double', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'name': 'threshold_y', 'edit_method': '', 'default': 0.002, 'level': 1, 'min': 0.001, 'type': 'double'}, {'srcline': 291, 'description': 'Threshold Z', 'max': 0.1, 'cconsttype': 'const double', 'ctype': 'double', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'name': 'threshold_z', 'edit_method': '', 'default': 0.002, 'level': 1, 'min': 0.001, 'type': 'double'}, {'srcline': 291, 'description': 'minimum_nr_points', 'max': 100.0, 'cconsttype': 'const double', 'ctype': 'double', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'name': 'minimum_nr_points', 'edit_method': '', 'default': 10.0, 'level': 1, 'min': 0.0, 'type': 'double'}, {'srcline': 291, 'description': 'Z Lower limit', 'max': 15.0, 'cconsttype': 'const double', 'ctype': 'double', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'name': 'z_lower_limit', 'edit_method': '', 'default': 0.0, 'level': 1, 'min': 0.0, 'type': 'double'}, {'srcline': 291, 'description': 'Z Upper limit', 'max': 50.0, 'cconsttype': 'const double', 'ctype': 'double', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'name': 'z_upper_limit', 'edit_method': '', 'default': 3.0, 'level': 1, 'min': 0.0, 'type': 'double'}, {'srcline': 291, 'description': 'X Lower limit', 'max': 0.0, 'cconsttype': 'const double', 'ctype': 'double', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'name': 'x_lower_limit', 'edit_method': '', 'default': -0.5, 'level': 1, 'min': -1.5, 'type': 'double'}, {'srcline': 291, 'description': 'X Upper limit', 'max': 1.5, 'cconsttype': 'const double', 'ctype': 'double', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'name': 'x_upper_limit', 'edit_method': '', 'default': 0.5, 'level': 1, 'min': 0.0, 'type': 'double'}, {'srcline': 291, 'description': 'Y Lower limit', 'max': 0.0, 'cconsttype': 'const double', 'ctype': 'double', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'name': 'y_lower_limit', 'edit_method': '', 'default': -0.5, 'level': 1, 'min': -1.5, 'type': 'double'}, {'srcline': 291, 'description': 'Y Upper limit', 'max': 1.5, 'cconsttype': 'const double', 'ctype': 'double', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'name': 'y_upper_limit', 'edit_method': '', 'default': 0.5, 'level': 1, 'min': 0.0, 'type': 'double'}, {'srcline': 291, 'description': 'Wrong_angle_threshold', 'max': 90.0, 'cconsttype': 'const double', 'ctype': 'double', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'name': 'angle_threshold', 'edit_method': '', 'default': 76.0, 'level': 1, 'min': 35.0, 'type': 'double'}, {'srcline': 291, 'description': 'Ground truth volume', 'max': 0.4, 'cconsttype': 'const double', 'ctype': 'double', 'srcfile': '/opt/ros/melodic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'name': 'real_volume', 'edit_method': '', 'default': 0.01689, 'level': 1, 'min': 0.01, 'type': 'double'}], 'type': '', 'id': 0}

min = {}
max = {}
defaults = {}
level = {}
type = {}
all_level = 0

#def extract_params(config):
#    params = []
#    params.extend(config['parameters'])
#    for group in config['groups']:
#        params.extend(extract_params(group))
#    return params

for param in extract_params(config_description):
    min[param['name']] = param['min']
    max[param['name']] = param['max']
    defaults[param['name']] = param['default']
    level[param['name']] = param['level']
    type[param['name']] = param['type']
    all_level = all_level | param['level']

