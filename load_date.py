import os
import requests
import tarfile
 
def download_and_extract_archive(url, destination_folder):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
 
    # Extract the file name from the URL to use as the local filename
    filename = os.path.join(destination_folder, url.split("/")[-1])
 
    # Download the archive
    response = requests.get(url, stream=True)
    with open(filename, 'wb') as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)
 
    # Extract the contents of the archive
    with tarfile.open(filename, 'r:gz') as tar:
        tar.extractall(destination_folder)
 
    # Remove the downloaded archive file
    os.remove(filename)
 
if __name__ == "__main__":
    archive_url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    destination_folder = "cifar100_dataset"  # Specify the folder where you want to extract the contents
 
    download_and_extract_archive(archive_url, destination_folder)
    print(f"Archive extracted to {destination_folder}")
