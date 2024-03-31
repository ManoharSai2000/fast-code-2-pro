from PIL import Image
import glob
import os


def convert_jpeg_to_jpg(input_file, output_file):
    try:
        # Open the JPEG image
        img = Image.open(input_file)
        # Save the image as JPG
        img.save(output_file, "JPEG", quality=100)
        print(
            "Conversion successful: {} converted to {}".format(input_file, output_file)
        )
    except Exception as e:
        print("Error during conversion: {}".format(e))


# Example usage
# input_file = "/afs/ece.cmu.edu/usr/aadeshkd/Private/workspace_646/Project/fast-code-2-pro/AlexNet/images_1/train/0/1.jpg"
# output_file = "output.jpeg"
# convert_jpeg_to_jpg(input_file, output_file)

path = "/home/ubuntu/fc2/train_transformed"

images = sorted(
    glob.glob(
        "/home/ubuntu/fc2/train_transformed/*"
    )
)
f = open("data.txt", "w")
for classe in [0]:
    files = sorted(glob.glob("/home/ubuntu/fc2/train_transformed/*"))

    out_path = (
        "/home/ubuntu/fc2/fast-code-2-pro/AlexNet/images/0"
        #+ classe.split("/")[-1]
        + "/"
    )
    #os.makedirs(out_path,exist_ok=True)
    for file in files:
        # line = str(classe[-1]) + " " + path + classe[-1] + "/" + file.split("/")[-1]
        # f.write(line + "\n")
        convert_jpeg_to_jpg(
            file, out_path + file.split("/")[-1].split(".")[0] + ".jpeg"
        )
