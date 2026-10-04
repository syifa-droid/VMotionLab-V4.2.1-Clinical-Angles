from .constants import DEFAULT_ANALYSIS_CONFIG, JOINTS, METHOD_OPTIONS

def normalize_side_selection(label):
    l=label.lower(); return ['left','right'] if l=='both' else [l]

def method_list_for_joint(config, joint):
    method=config.get('marker_methods',{}).get(joint,'direct')
    return ['direct','virtual'] if method=='compare' else [method]

def side_suffix(side): return 'l' if side=='left' else 'r'
