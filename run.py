from google.colab import drive # import drive from google colab

ROOT = "/content/drive"     # default location for the drive
print(ROOT)                 # print content of ROOT (Optional)

drive.mount(ROOT)           # we mount the google drive at /content/drive
# Clone github repository setup
# import join used to join ROOT path and MY_GOOGLE_DRIVE_PATH
from os.path import join

# path to your project on Google Drive
MY_GOOGLE_DRIVE_PATH = 'My Drive/MyDrive/Udacity/deep-learning-v2-pytorch'
# replace with your Github username
GIT_USERNAME = "vsay01"
# definitely replace with your
GIT_TOKEN = "{YOUR_GITHUB_TOKEN}"
# Replace with your github repository in this case we want
# to clone deep-learning-v2-pytorch repository
GIT_REPOSITORY = "deep-learning-v2-pytorch"

PROJECT_PATH = join(ROOT, MY_GOOGLE_DRIVE_PATH)

# It's good to print out the value if you are not sure
print("PROJECT_PATH: ", PROJECT_PATH)

# In case we haven't created the folder already; we will create a folder in the project path
!mkdir "{PROJECT_PATH}"

#GIT_PATH = "https://{GIT_TOKEN}@github.com/{GIT_USERNAME}/{GIT_REPOSITORY}.git" this return 400 Bad Request for me
GIT_PATH = "https://" + GIT_TOKEN + "@github.com/" + GIT_USERNAME + "/" + GIT_REPOSITORY + ".git"
print("GIT_PATH: ", GIT_PATH)