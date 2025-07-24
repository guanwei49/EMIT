# string_list = ['Where is the defect?', 'Is there any defect in the object?', 'What is the type of the defect?',

string_list = ['Where is the defect?', 'Is there any defect in the object?']

dataset_list = ['MVTec','VisA']

def if_use_comp(target_string, dataset_r = 'MVTec'):
    if dataset_r not in dataset_list:
        return False

    for s in string_list:
        if s in target_string:
            return True
    return False
