import os

def explore_files(parent_folder):
    # Create a dictionary of the classes
    classes_dict = {}
    annotations_paths = set()
    for root, dirs, files in os.walk(parent_folder):
        for file in files:

            if file.endswith(".txt") and file[0].islower():
                file_path = os.path.join(root, file)
                # Do something with the file_path (e.g., print, process, etc.)
                file_name = file.split('.')[0]
                class_name = file_name.split('_')[0]

                if root.split('\\')[-1] == 'Annotations':
                    root = root.replace('\\', '/')
                    parts = root.split('/')
                    annotations_paths.add(parts[-3] + '/' + parts[-2] + '/' + parts[-1])

                if class_name in classes_dict:
                    classes_dict[class_name].append(file_name)
                else:
                    classes_dict[class_name] = [file_name]

                print(file_name, class_name)

    return classes_dict, annotations_paths

# Example usage
parent_folder = "path/to/parent/folder"
explore_files(parent_folder)

# Return number of classes

def main():
    # Create anno_paths.txt
    classes_dict, annotations_paths = explore_files('./data/s3dis/Stanford3dDataset_v1.2_Aligned_Version')
    print(classes_dict.keys())

    annotations_paths = sorted(annotations_paths)

    with open('./data_utils/meta/anno_paths.txt', 'w') as file:
        for item in annotations_paths:
            file.write(item + '\n')
            print(item)

    with open('./data_utils/meta/class_names.txt', 'w') as file:
        for item in classes_dict.keys():
            file.write(item + '\n')
            print(item)


    # Create class_names.txt

if __name__ == '__main__':
    main()