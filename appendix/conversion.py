import os
import shutil

directory = "./raw-mtn"
new_header = "timestamps,acc_x,acc_y,acc_z\n"
destination_directory = "./mtn"

for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    

    if os.path.isfile(file_path):

        with open(file_path, 'r') as f:
            lines = f.readlines()[3:]
        

        with open(file_path, 'w') as f:
            f.write(new_header)
            f.writelines(lines)  

        shutil.copy(file_path, os.path.join(destination_directory, filename))