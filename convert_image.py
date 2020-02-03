import mritopng
import os

folder_path = r"D:\FYP\imageConvert\dicom_image\C3N-01944\09-06-2000-Szyja i krtan z kontrastem-98323\5-SZYJACM  1.0  I26s  3-70627"
destinate_path = 'D:/FYP/imageConvert/dicom_image/C3N-01944/09-06-2000-Szyja i krtan z kontrastem-98323/5-SZYJACM  1.0  I26s  3-70627/PNG/'

# def convertImg(dicomImage):
#     print("Image: " + dicomImage)
#     outputImg = dicomImage.split('.')
#     mritopng.convert_file(os.path.join(folder_path, dicomImage), os.path.join(folder_path, outputImg[0] + '.jpg'), auto_contrast=True)

#     convertedImg = os.path.join(folder_path, outputImg[0] + '.jpg')
#     return convertedImg

# mritopng.convert_folder(folder_path, destinate_path)

def main(): 
    i = 0

    for filename in os.listdir(folder_path): 
        dst =str(i) + ".png"
        src =destinate_path + filename + ".png"
        dst =destinate_path + dst 
          
        # rename() function will 
        # rename all the files 
        os.rename(src, dst) 
        i += 1
  
# Driver Code 
if __name__ == '__main__': 
      
    # Calling main() function 
    main() 