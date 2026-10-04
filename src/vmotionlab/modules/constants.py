from __future__ import annotations

HALPE26 = {
    'Nose':0, 'LShoulder':5, 'RShoulder':6, 'LElbow':7, 'RElbow':8,
    'LWrist':9, 'RWrist':10, 'LHip':11, 'RHip':12, 'LKnee':13,
    'RKnee':14, 'LAnkle':15, 'RAnkle':16, 'Head':17, 'Neck':18,
    'Hip':19, 'LBigToe':20, 'RBigToe':21, 'LSmallToe':22,
    'RSmallToe':23, 'LHeel':24, 'RHeel':25,
}
SIDE = {
 'left': {'shoulder':'LShoulder','hip':'LHip','knee':'LKnee','ankle':'LAnkle','heel':'LHeel','big_toe':'LBigToe','small_toe':'LSmallToe'},
 'right': {'shoulder':'RShoulder','hip':'RHip','knee':'RKnee','ankle':'RAnkle','heel':'RHeel','big_toe':'RBigToe','small_toe':'RSmallToe'},
}
JOINTS=['hip','knee','ankle']
METHOD_OPTIONS={'Direct':'direct','Virtual':'virtual','Compare':'compare'}
DEFAULT_ANALYSIS_CONFIG={
 'selected_sides':['right'], 'selected_joints':['hip','knee','ankle'],
 'marker_methods':{'hip':'direct','knee':'direct','ankle':'direct'},
 'quality':{'min_visibility':0.40,'min_presence':0.40,'max_normalized_jump':0.10,'max_short_gap_frames':10},
 'virtual_markers':{'reference_percentile':95.0},
 'filter':{'cutoff_hz':6.0,'order':4},
}
DEFAULT_COM_CONFIG={
 'anthropometric_model':'average','min_visibility':0.4,'min_presence':0.4,
 'min_valid_mass_fraction':0.70,'filter_cutoff_hz':3.0,'filter_order':4,
}
