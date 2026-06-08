import glob
import os


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def clean_pt_files_in_dir(path_to_dir):
    filelist = glob.glob(os.path.join(path_to_dir, "*.pth"))
    for file_path in filelist:
        os.remove(file_path)


def sort_file_by_digit_in_name(filelist, suffix=".pth"):
    end_index = len(suffix)
    file_name_list_int = [int(os.path.basename(file_path)[:-end_index]) for file_path in filelist]
    folder_name = os.path.dirname(filelist[0])
    sorted_file_name_list_int = sorted(file_name_list_int)
    return [
        os.path.join(folder_name, "{}{}".format(num, suffix))
        for num in sorted_file_name_list_int
    ]


def num_list_to_str(num_list):
    return "".join(str(i) for i in num_list)
