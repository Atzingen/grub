'''
curl -L "https://app.roboflow.com/ds/DWJONAU3Gc?key=IqgJV2tJ4l" &gt; roboflow.zip; unzip roboflow.zip; rm roboflow.zip

https://app.roboflow.com/ds/DWJONAU3Gc?key=IqgJV2tJ4l
                
'''
import os
from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()

api_key = os.getenv("ROBOFLOW_API_KEY")
rf = Roboflow(api_key=api_key)
project = rf.workspace("gustavo-voltani-von-atzingen").project("grobs")
version = project.version(5)
dataset = version.download("yolov8")

