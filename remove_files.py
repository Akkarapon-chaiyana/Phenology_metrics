import os
import shutil

def remove_files_in_folder(folder_path):
    """
    Removes all files and directories within the specified folder.

    Args:
        folder_path (str): The path to the folder whose contents should be removed.

    Returns:
        None
    """
    # Ensure the folder exists before proceeding
    if os.path.exists(folder_path):
        # Remove all files in the folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                # Check if it is a file or a directory and delete accordingly
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove the file
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove the directory
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
        print(f'All contents of the folder {folder_path} have been successfully removed!!!')
    else:
        print(f'The folder {folder_path} does not exist')
